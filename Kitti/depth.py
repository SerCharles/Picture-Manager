from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
import PIL
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import cv2 as cv
import os
import random

# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

alpha_recon_image = 0.85

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def flow_warp(src_img, flow):
  """ inverse warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
  """
  batch, height, width, _ = src_img.get_shape().as_list()
  tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                     [0, 2, 3, 1])
  src_pixel_coords = tgt_pixel_coords + flow
  output_img = bilinear_sampler(src_img, src_pixel_coords)
  return output_img

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
  """Compute the rigid flow from target image plane to source image

  Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True) 
          camera transformation matrix [batch, 6], in the order of 
          tx, ty, tz, rx, ry, rz; 
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
  """
  batch, height, width = depth.get_shape().as_list()
  # Convert pose vector to matrix
  pose = pose_vec2mat(pose)
  if reverse_pose:
    pose = tf.matrix_inverse(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  tgt_pixel_coords = tf.transpose(pixel_coords[:,:2,:,:], [0, 2, 3, 1])
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  rigid_flow = src_pixel_coords - tgt_pixel_coords
  return rigid_flow

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output

def forward_warp(imgs, coords):
  """Construct a new image by warp input image to a new perspective.

  points out of the boundary are discarded.

  Args:
    imgs: source image to be warped [height_s, width_s, channels]
    coords: coordinates of source pixels to warp to [height_s,
      width_s, 3]. the last dimension is (x,y,d)
  Returns:
    A warped image [height_s, width_s, channels] and a mask where valid
    pixels are assigned true.
  """
  height, width, channels = imgs.get_shape().as_list()
  pix_num = width * height

  # concate src index and tgt index
  meshgrid = tf.meshgrid(tf.range(height), tf.range(width), indexing = 'ij')
  meshgrid = tf.cast(meshgrid, tf.float32)
  meshgrid = tf.transpose(meshgrid, [1,2,0])

  # last dim: src_x, src_y, tgt_x, tgt_y, depth
  map_data_meshgrid = tf.concat([meshgrid, coords], axis = 2)
  map_data_list = tf.reshape(map_data_meshgrid, [-1, 5])

  # sort 3 passes by d, tgt_x, tgt_y
  # d
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-1]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # y
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-2]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # x
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-3]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)

  # mask out the pixels to be discarded
  mask = tf.equal(tf.cast(tf.floor(map_data_list[1:,2:4]), dtype=tf.int32), tf.cast(tf.floor(map_data_list[:-1,2:4]), dtype=tf.int32))
  mask = tf.logical_and(mask[:,0], mask[:,1])
  mask = tf.cast(tf.concat([[False], mask], axis=0), tf.float32)

  map_data_list = tf.concat([map_data_list[:,:4], tf.expand_dims(mask, 1)], axis=-1)

  # sort 2 passes by src_x, src_y
  # y
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-map_data_list[:,1], pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # x
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-map_data_list[:,0], pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)

  # discard the ocluded area
  mask_2d = tf.reshape(map_data_list[:,4], [height, width])
  non_overlapping_coords = tf.reshape(map_data_list[:,2:4], [height, width, 2])
  bound = tf.expand_dims([height * 1.0, width * 1.0], 0)
  bound = tf.expand_dims(bound, 0)
  bound = tf.tile(bound + 1, [height, width, 1])
  non_overlapping_coords = tf.where(tf.tile(tf.expand_dims(tf.cast(mask_2d, tf.bool),2),[1,1,2]), \
  bound, non_overlapping_coords)
  non_overlapping_coords = tf.concat([\
  tf.clip_by_value(non_overlapping_coords[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords[:,:,1:], -2, width + 1)], axis=2)

  # discard, because interpolate is necessary
  #non_overlapping_coords = tf.cast(tf.floor(non_overlapping_coords), tf.int32)

  non_overlapping_coords00 = tf.floor(non_overlapping_coords)
  non_overlapping_coords01 = tf.concat([non_overlapping_coords00[:,:,:1], non_overlapping_coords00[:,:,1:] + 1], axis=-1)
  non_overlapping_coords01 = tf.concat([\
  tf.clip_by_value(non_overlapping_coords01[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords01[:,:,1:], -2, width + 1)], axis=2)
  non_overlapping_coords10 = tf.concat([non_overlapping_coords00[:,:,:1] + 1, non_overlapping_coords00[:,:,1:]], axis=-1)
  non_overlapping_coords10 = tf.concat([\
  tf.clip_by_value(non_overlapping_coords10[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords10[:,:,1:], -2, width + 1)], axis=2)
  non_overlapping_coords11 = tf.concat([non_overlapping_coords10[:,:,:1], non_overlapping_coords01[:,:,1:]], axis=-1)

  # distance00 = 1.0 / (tf.norm(non_overlapping_coords00 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance01 = 1.0 / (tf.norm(non_overlapping_coords01 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance10 = 1.0 / (tf.norm(non_overlapping_coords10 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance11 = 1.0 / (tf.norm(non_overlapping_coords11 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))

  distance00 = non_overlapping_coords00 - non_overlapping_coords
  distance01 = non_overlapping_coords01 - non_overlapping_coords
  distance10 = non_overlapping_coords10 - non_overlapping_coords
  distance11 = non_overlapping_coords11 - non_overlapping_coords

  # area00 = distance11[:,:,:1] * distance11[:,:,1:]
  # area01 = -distance10[:,:,:1] * distance10[:,:,1:]
  # area10 = -distance01[:,:,:1] * distance01[:,:,1:]
  # area11 = distance00[:,:,:1] * distance00[:,:,1:]

  area11 = 1 - distance11[:,:,:1] * distance11[:,:,1:]
  area10 = 1 + distance10[:,:,:1] * distance10[:,:,1:]
  area01 = 1 + distance01[:,:,:1] * distance01[:,:,1:]
  area00 = 1 - distance00[:,:,:1] * distance00[:,:,1:]

  # forward warping (warp coords to pass the flow value)
  result00 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords00 + 2, tf.int32), updates=tf.concat([imgs, area00, coords], axis=2)\
    , shape=[height + 4, width + 4, channels + 4])
  result01 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords01 + 2, tf.int32), updates=tf.concat([imgs, area01], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])
  result10 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords10 + 2, tf.int32), updates=tf.concat([imgs, area10], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])
  result11 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords11 + 2, tf.int32), updates=tf.concat([imgs, area11], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])



  weight00 = tf.tile(result00[2:-2,2:-2,3:4], [1,1,3])
  weight01 = tf.tile(result01[2:-2,2:-2,3:4], [1,1,3])
  weight10 = tf.tile(result10[2:-2,2:-2,3:4], [1,1,3])
  weight11 = tf.tile(result11[2:-2,2:-2,3:4], [1,1,3])

  weight_sum = tf.stop_gradient(weight00 + weight01 + weight10 + weight11)

  result_sum =  result00[2:-2,2:-2,:3] * weight00 + \
          result01[2:-2,2:-2,:3] * weight01 + \
          result10[2:-2,2:-2,:3] * weight10 + \
          result11[2:-2,2:-2,:3] * weight11

  result = result_sum / (weight_sum + 0.00001)
  mask = tf.cast(weight_sum, tf.bool)
  mask = tf.cast(mask, tf.float32)
  warped_coords = result00[2:-2,2:-2,4:]

  return result, mask, warped_coords

def f2f(image):
  image = tf.cast(image, dtype=tf.bool)
  image = tf.cast(image, dtype=tf.float32)
  return image

def forward_flow_warp(src_img, flow, depth):
  """ forward warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
    depth: depth map of the src_img
  Returns:
    Source image forward warped to the target image plane [batch, height_t, width_t, 3]
  """
  batch, height, width, _ = src_img.get_shape().as_list()
  tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                     [0, 2, 3, 1]) + flow
  tgt_pixel_coords = tgt_pixel_coords[:,:,:,::-1]
  tgt_pixel_coords_depth = tf.concat([tgt_pixel_coords, depth], axis=-1)
  for i in range(batch):
    warped, mask, coords = forward_warp(src_img[i,:,:,:], tgt_pixel_coords_depth[i,:,:,:])
    if i == 0:
      result = tf.expand_dims(warped, 0)
      mask_r = tf.expand_dims(mask, 0)
      coords_r = tf.expand_dims(coords, 0)
    else:
      result = tf.concat([result, tf.expand_dims(warped, 0)], axis=0)
      mask_r = tf.concat([mask_r, tf.expand_dims(mask, 0)], axis=0)
      coords_r = tf.concat([coords_r, tf.expand_dims(coords, 0)], axis=0)
  return result, mask_r, coords_r

def preprocess_image(image):
    # Assuming input image is uint8
    if image == None:
        return None
    else:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

def deprocess_image(image):
    # Assuming input image is float32
    image = (image + 1.)/2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def image_similarity(x, y):
    #return tf.abs(x-y)
    return alpha_recon_image * SSIM(x, y) + (1-alpha_recon_image) * tf.abs(x-y)

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def compute_smooth_loss(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

def scale_2_255(img, type):
  if (type == 'depth'):
    return (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
  elif (type == 'warped'):
    return (img + 1) * 128
  elif (type == '200'):
    return (img + 1) * 100

def gray2color(img):
  return tf.tile(img, [1,1,1,3])

def spatial_normalize(disp):
  _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
  disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
  disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
  return disp/disp_mean

def down_sample(image, scale = 2):
  shape = image.get_shape().as_list()
  image = tf.image.resize_images(image, \
      [int(shape[1] / scale), int(shape[2] / scale)], method=tf.image.ResizeMethod.BILINEAR)
  return image

def L2_norm(x, axis=3, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
    return l2_norm
# The network design is based on Tinghui Zhou & Clement Godard's works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_model.pyimport tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import cv2 as cv
import os
import random

# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

alpha_recon_image = 0.85

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = vec.get_shape().as_list()
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """
  batch, height, width = depth.get_shape().as_list()
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height*width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords

def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """
  batch, _, height, width = cam_coords.get_shape().as_list()
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis=0)
  else:
    coords = tf.stack([x_t, y_t], axis=0)
  coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
  return coords

def flow_warp(src_img, flow):
  """ inverse warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
  Returns:
    Source image inverse warped to the target image plane [batch, height_t, width_t, 3]
  """
  batch, height, width, _ = src_img.get_shape().as_list()
  tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                     [0, 2, 3, 1])
  src_pixel_coords = tgt_pixel_coords + flow
  output_img = bilinear_sampler(src_img, src_pixel_coords)
  return output_img

def compute_rigid_flow(depth, pose, intrinsics, reverse_pose=False):
  """Compute the rigid flow from target image plane to source image

  Args:
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source (or source to target if reverse_pose=True) 
          camera transformation matrix [batch, 6], in the order of 
          tx, ty, tz, rx, ry, rz; 
    intrinsics: camera intrinsics [batch, 3, 3]
  Returns:
    Rigid flow from target image to source image [batch, height_t, width_t, 2]
  """
  batch, height, width = depth.get_shape().as_list()
  # Convert pose vector to matrix
  pose = pose_vec2mat(pose)
  if reverse_pose:
    pose = tf.matrix_inverse(pose)
  # Construct pixel grid coordinates
  pixel_coords = meshgrid(batch, height, width)
  tgt_pixel_coords = tf.transpose(pixel_coords[:,:2,:,:], [0, 2, 3, 1])
  # Convert pixel coordinates to the camera frame
  cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
  # Construct a 4x4 intrinsic matrix
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch, 1, 1])
  intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
  intrinsics = tf.concat([intrinsics, filler], axis=1)
  # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
  # pixel frame.
  proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
  src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
  rigid_flow = src_pixel_coords - tgt_pixel_coords
  return rigid_flow

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output

def forward_warp(imgs, coords):
  """Construct a new image by warp input image to a new perspective.

  points out of the boundary are discarded.

  Args:
    imgs: source image to be warped [height_s, width_s, channels]
    coords: coordinates of source pixels to warp to [height_s,
      width_s, 3]. the last dimension is (x,y,d)
  Returns:
    A warped image [height_s, width_s, channels] and a mask where valid
    pixels are assigned true.
  """
  height, width, channels = imgs.get_shape().as_list()
  pix_num = width * height

  # concate src index and tgt index
  meshgrid = tf.meshgrid(tf.range(height), tf.range(width), indexing = 'ij')
  meshgrid = tf.cast(meshgrid, tf.float32)
  meshgrid = tf.transpose(meshgrid, [1,2,0])

  # last dim: src_x, src_y, tgt_x, tgt_y, depth
  map_data_meshgrid = tf.concat([meshgrid, coords], axis = 2)
  map_data_list = tf.reshape(map_data_meshgrid, [-1, 5])

  # sort 3 passes by d, tgt_x, tgt_y
  # d
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-1]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # y
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-2]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # x
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-tf.floor(map_data_list[:,-3]), pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)

  # mask out the pixels to be discarded
  mask = tf.equal(tf.cast(tf.floor(map_data_list[1:,2:4]), dtype=tf.int32), tf.cast(tf.floor(map_data_list[:-1,2:4]), dtype=tf.int32))
  mask = tf.logical_and(mask[:,0], mask[:,1])
  mask = tf.cast(tf.concat([[False], mask], axis=0), tf.float32)

  map_data_list = tf.concat([map_data_list[:,:4], tf.expand_dims(mask, 1)], axis=-1)

  # sort 2 passes by src_x, src_y
  # y
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-map_data_list[:,1], pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)
  # x
  sorted_d_index = tf.expand_dims(tf.nn.top_k(-map_data_list[:,0], pix_num).indices, 1)
  map_data_list = tf.gather_nd(map_data_list, sorted_d_index)

  # discard the ocluded area
  mask_2d = tf.reshape(map_data_list[:,4], [height, width])
  non_overlapping_coords = tf.reshape(map_data_list[:,2:4], [height, width, 2])
  bound = tf.expand_dims([height * 1.0, width * 1.0], 0)
  bound = tf.expand_dims(bound, 0)
  bound = tf.tile(bound + 1, [height, width, 1])
  non_overlapping_coords = tf.where(tf.tile(tf.expand_dims(tf.cast(mask_2d, tf.bool),2),[1,1,2]), \
  bound, non_overlapping_coords)
  non_overlapping_coords = tf.concat([\
  tf.clip_by_value(non_overlapping_coords[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords[:,:,1:], -2, width + 1)], axis=2)

  # discard, because interpolate is necessary
  #non_overlapping_coords = tf.cast(tf.floor(non_overlapping_coords), tf.int32)

  non_overlapping_coords00 = tf.floor(non_overlapping_coords)
  non_overlapping_coords01 = tf.concat([non_overlapping_coords00[:,:,:1], non_overlapping_coords00[:,:,1:] + 1], axis=-1)
  non_overlapping_coords01 = tf.concat([\
  tf.clip_by_value(non_overlapping_coords01[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords01[:,:,1:], -2, width + 1)], axis=2)
  non_overlapping_coords10 = tf.concat([non_overlapping_coords00[:,:,:1] + 1, non_overlapping_coords00[:,:,1:]], axis=-1)
  non_overlapping_coords10 = tf.concat([\
  tf.clip_by_value(non_overlapping_coords10[:,:,:1], -2, height + 1),\
  tf.clip_by_value(non_overlapping_coords10[:,:,1:], -2, width + 1)], axis=2)
  non_overlapping_coords11 = tf.concat([non_overlapping_coords10[:,:,:1], non_overlapping_coords01[:,:,1:]], axis=-1)

  # distance00 = 1.0 / (tf.norm(non_overlapping_coords00 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance01 = 1.0 / (tf.norm(non_overlapping_coords01 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance10 = 1.0 / (tf.norm(non_overlapping_coords10 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))
  # distance11 = 1.0 / (tf.norm(non_overlapping_coords11 - non_overlapping_coords + 0.001, axis=-1, keep_dims = True))

  distance00 = non_overlapping_coords00 - non_overlapping_coords
  distance01 = non_overlapping_coords01 - non_overlapping_coords
  distance10 = non_overlapping_coords10 - non_overlapping_coords
  distance11 = non_overlapping_coords11 - non_overlapping_coords

  # area00 = distance11[:,:,:1] * distance11[:,:,1:]
  # area01 = -distance10[:,:,:1] * distance10[:,:,1:]
  # area10 = -distance01[:,:,:1] * distance01[:,:,1:]
  # area11 = distance00[:,:,:1] * distance00[:,:,1:]

  area11 = 1 - distance11[:,:,:1] * distance11[:,:,1:]
  area10 = 1 + distance10[:,:,:1] * distance10[:,:,1:]
  area01 = 1 + distance01[:,:,:1] * distance01[:,:,1:]
  area00 = 1 - distance00[:,:,:1] * distance00[:,:,1:]

  # forward warping (warp coords to pass the flow value)
  result00 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords00 + 2, tf.int32), updates=tf.concat([imgs, area00, coords], axis=2)\
    , shape=[height + 4, width + 4, channels + 4])
  result01 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords01 + 2, tf.int32), updates=tf.concat([imgs, area01], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])
  result10 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords10 + 2, tf.int32), updates=tf.concat([imgs, area10], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])
  result11 = tf.scatter_nd(indices=tf.cast(non_overlapping_coords11 + 2, tf.int32), updates=tf.concat([imgs, area11], axis=2)\
    , shape=[height + 4, width + 4, channels + 1])



  weight00 = tf.tile(result00[2:-2,2:-2,3:4], [1,1,3])
  weight01 = tf.tile(result01[2:-2,2:-2,3:4], [1,1,3])
  weight10 = tf.tile(result10[2:-2,2:-2,3:4], [1,1,3])
  weight11 = tf.tile(result11[2:-2,2:-2,3:4], [1,1,3])

  weight_sum = tf.stop_gradient(weight00 + weight01 + weight10 + weight11)

  result_sum =  result00[2:-2,2:-2,:3] * weight00 + \
          result01[2:-2,2:-2,:3] * weight01 + \
          result10[2:-2,2:-2,:3] * weight10 + \
          result11[2:-2,2:-2,:3] * weight11

  result = result_sum / (weight_sum + 0.00001)
  mask = tf.cast(weight_sum, tf.bool)
  mask = tf.cast(mask, tf.float32)
  warped_coords = result00[2:-2,2:-2,4:]

  return result, mask, warped_coords

def f2f(image):
  image = tf.cast(image, dtype=tf.bool)
  image = tf.cast(image, dtype=tf.float32)
  return image

def forward_flow_warp(src_img, flow, depth):
  """ forward warp a source image to the target image plane based on flow field
  Args:
    src_img: the source  image [batch, height_s, width_s, 3]
    flow: target image to source image flow [batch, height_t, width_t, 2]
    depth: depth map of the src_img
  Returns:
    Source image forward warped to the target image plane [batch, height_t, width_t, 3]
  """
  batch, height, width, _ = src_img.get_shape().as_list()
  tgt_pixel_coords = tf.transpose(meshgrid(batch, height, width, False),
                     [0, 2, 3, 1]) + flow
  tgt_pixel_coords = tgt_pixel_coords[:,:,:,::-1]
  tgt_pixel_coords_depth = tf.concat([tgt_pixel_coords, depth], axis=-1)
  for i in range(batch):
    warped, mask, coords = forward_warp(src_img[i,:,:,:], tgt_pixel_coords_depth[i,:,:,:])
    if i == 0:
      result = tf.expand_dims(warped, 0)
      mask_r = tf.expand_dims(mask, 0)
      coords_r = tf.expand_dims(coords, 0)
    else:
      result = tf.concat([result, tf.expand_dims(warped, 0)], axis=0)
      mask_r = tf.concat([mask_r, tf.expand_dims(mask, 0)], axis=0)
      coords_r = tf.concat([coords_r, tf.expand_dims(coords, 0)], axis=0)
  return result, mask_r, coords_r

def preprocess_image(image):
    # Assuming input image is uint8
    if image == None:
        return None
    else:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

def deprocess_image(image):
    # Assuming input image is float32
    image = (image + 1.)/2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def image_similarity(x, y):
    #return tf.abs(x-y)
    return alpha_recon_image * SSIM(x, y) + (1-alpha_recon_image) * tf.abs(x-y)

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def compute_smooth_loss(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

def scale_2_255(img, type):
  if (type == 'depth'):
    return (img - np.min(img)) * 255 / (np.max(img) - np.min(img))
  elif (type == 'warped'):
    return (img + 1) * 128
  elif (type == '200'):
    return (img + 1) * 100

def gray2color(img):
  return tf.tile(img, [1,1,1,3])

def spatial_normalize(disp):
  _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
  disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
  disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
  return disp/disp_mean

def down_sample(image, scale = 2):
  shape = image.get_shape().as_list()
  image = tf.image.resize_images(image, \
      [int(shape[1] / scale), int(shape[2] / scale)], method=tf.image.ResizeMethod.BILINEAR)
  return image

def L2_norm(x, axis=3, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
    return l2_norm
# The network design is based on Tinghui Zhou & Clement Godard's works:
# https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING_RESNET50 = 5
DISP_SCALING_VGG = 10
FLOW_SCALING = 0.1

def color_network(image, mask, vscope='cnet_d', reuse_weights=tf.AUTO_REUSE):
    isTrainable = opt.train_color_net
    return build_resnet50(tf.concat([image, mask], axis=-1), get_res_resnet50, isTrainable, vscope)

def disp_net(opt, dispnet_inputs):
    is_training = opt.train_disp_net
    return build_resnet50(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net')

def flow_net(opt, flownet_inputs):
    is_training = (opt.add_flow_loss and opt.train_flow_net)
    return build_resnet50(flownet_inputs, get_flow, is_training, 'flow_net')

def pose_net(opt, posenet_inputs):
    is_training = opt.mode == 'train'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('pose_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1  = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2  = slim.conv2d(conv1, 32,  5, 2)
            conv3  = slim.conv2d(conv2, 64,  3, 2)
            conv4  = slim.conv2d(conv3, 128, 3, 2)
            conv5  = slim.conv2d(conv4, 256, 3, 2)
            conv6  = slim.conv2d(conv5, 256, 3, 2)
            conv7  = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 12, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, 2, 6])
            return pose_final

def critic(opt, image):
    is_training = opt.mode == 'train'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as sc:
        with slim.arg_scope([slim.conv2d],
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params,
                    weights_regularizer=slim.l2_regularizer(0.0001),
                    activation_fn=tf.nn.leaky_relu):
            size = 64
            conv1  = slim.conv2d(image, size    ,  3, 2)
            conv2  = slim.conv2d(conv1, size * 2,  3, 2)
            conv3  = slim.conv2d(conv2, size * 4,  3, 2)
            conv4  = slim.conv2d(conv3, size * 8,  3, 2)
            ret    = slim.fully_connected(conv3, 1, activation_fn=None)
            return ret

def build_resnet50(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = conv(inputs, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # DECODING
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_pred(iconv4)
            upred4  = upsample_nn(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_pred(iconv3)
            upred3  = upsample_nn(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_pred(iconv2)
            upred2  = upsample_nn(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_pred(iconv1)

            return pred1

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def get_disp_vgg(x):
    disp = DISP_SCALING_VGG * slim.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_res_resnet50(x):
    return DISP_SCALING_RESNET50 * conv(x, 3, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01

def get_flow(x):
    # Output flow value is normalized by image height/width
    flow = FLOW_SCALING * slim.conv2d(x, 2, 3, 1, activation_fn=None, normalizer_fn=None)
    return flow

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv

def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

class depth_predictor(object):
    def __init__(self, opt=None):
        self.opt = opt
        
    def build_model(self, pre_image_f, tgt_image_f, aft_image_f, intrinsics):
        opt = self.opt

        # pred_disp_pre = disp_net(opt, pre_image_f)
        # pred_disp_tgt = disp_net(opt, tgt_image_f)
        # pred_disp_aft = disp_net(opt, aft_image_f)

        pred_disp_pre = spatial_normalize(disp_net(opt, pre_image_f))
        pred_disp_tgt = spatial_normalize(disp_net(opt, tgt_image_f))
        pred_disp_aft = spatial_normalize(disp_net(opt, aft_image_f))

        pred_depth_pre = 1. / pred_disp_pre
        pred_depth_tgt = 1. / pred_disp_tgt
        pred_depth_aft = 1. / pred_disp_aft

        if intrinsics == None:
            self.pred_depth_tgt = pred_depth_tgt
            return

        pred_poses = pose_net(opt, tf.concat([tgt_image_f, pre_image_f, aft_image_f], axis=3))

        tgt2pre_flow = compute_rigid_flow(pred_depth_tgt[:,:,:,0], pred_poses[:,0,:], intrinsics, False)
        tgt2aft_flow = compute_rigid_flow(pred_depth_tgt[:,:,:,0], pred_poses[:,1,:], intrinsics, False)
        pre2tgt_flow = compute_rigid_flow(pred_depth_pre[:,:,:,0], pred_poses[:,0,:], intrinsics, True)
        aft2tgt_flow = compute_rigid_flow(pred_depth_aft[:,:,:,0], pred_poses[:,1,:], intrinsics, True)

        tgt2pre_mask = tf.ones_like(pre_image_f)
        tgt2aft_mask = tf.ones_like(aft_image_f)
        aft2tgt_mask = tf.ones_like(tgt_image_f)
        pre2tgt_mask = tf.ones_like(tgt_image_f)

        if opt.use_forward_warp:
            # forward 
            tgt2pre_warp, tgt2pre_mask, tgt2pre_coords = forward_flow_warp(tgt_image_f, tgt2pre_flow, pred_depth_tgt)
            tgt2aft_warp, tgt2aft_mask, tgt2aft_coords = forward_flow_warp(tgt_image_f, tgt2aft_flow, pred_depth_tgt)
            aft2tgt_warp, aft2tgt_mask, aft2tgt_coords = forward_flow_warp(aft_image_f, aft2tgt_flow, pred_depth_aft)
            pre2tgt_warp, pre2tgt_mask, pre2tgt_coords = forward_flow_warp(pre_image_f, pre2tgt_flow, pred_depth_pre)

        else:
            # backward
            tgt2pre_warp = flow_warp(tgt_image_f, pre2tgt_flow)
            tgt2aft_warp = flow_warp(tgt_image_f, aft2tgt_flow)
            aft2tgt_warp = flow_warp(aft_image_f, tgt2aft_flow)
            pre2tgt_warp = flow_warp(pre_image_f, tgt2pre_flow)

            tgt2pre_mask = pred_depth_pre
            tgt2aft_mask = pred_depth_aft
            aft2tgt_mask = pred_depth_tgt
            pre2tgt_mask = pred_depth_tgt

        if opt.use_color_net:
            tgt2pre_warp = color_network(tgt2pre_warp, tgt2pre_mask)
            tgt2aft_warp = color_network(tgt2aft_warp, tgt2aft_mask)
            aft2tgt_warp = color_network(aft2tgt_warp, aft2tgt_mask)
            pre2tgt_warp = color_network(pre2tgt_warp, pre2tgt_mask)

        aft2tgt_error = image_similarity(aft2tgt_warp, tgt_image_f * aft2tgt_mask)
        pre2tgt_error = image_similarity(pre2tgt_warp, tgt_image_f * pre2tgt_mask)
        tgt2aft_error = image_similarity(tgt2aft_warp, aft_image_f * tgt2aft_mask)
        tgt2pre_error = image_similarity(tgt2pre_warp, pre_image_f * tgt2pre_mask)

        tgt_error = tf.reduce_mean(aft2tgt_error)
        tgt_error +=tf.reduce_mean(pre2tgt_error)
        aft_error = tf.reduce_mean(image_similarity(tgt2aft_warp, aft_image_f))
        pre_error = tf.reduce_mean(image_similarity(tgt2pre_warp, pre_image_f))

        # tgt_error = tf.reduce_sum(image_similarity(aft2tgt_warp, tgt_image_f * aft2tgt_mask)) / tf.reduce_sum(aft2tgt_mask)
        # tgt_error +=tf.reduce_sum(image_similarity(pre2tgt_warp, tgt_image_f * pre2tgt_mask)) / tf.reduce_sum(pre2tgt_mask)
        # aft_error = tf.reduce_sum(image_similarity(tgt2aft_warp, aft_image_f * tgt2aft_mask)) / tf.reduce_sum(tgt2aft_mask)
        # pre_error = tf.reduce_sum(image_similarity(tgt2pre_warp, pre_image_f * tgt2pre_mask)) / tf.reduce_sum(tgt2pre_mask)

        tgt_loss = tgt_error
        src_loss = aft_error + pre_error
        smooth_loss = 0.1 * (compute_smooth_loss(pred_disp_pre, pre_image_f) + compute_smooth_loss(pred_disp_aft, aft_image_f) + compute_smooth_loss(pred_disp_tgt, tgt_image_f))
        loss = tf.reduce_mean(tgt2pre_error) + tf.reduce_mean(aft2tgt_error)
        loss = tf.reduce_mean(pre2tgt_error) + tf.reduce_mean(tgt2aft_error)

        self.loss = loss
        self.tgt_loss = tgt_loss
        self.src_loss = src_loss
        self.smooth_loss = smooth_loss
        self.tgt2pre_warp = tgt2pre_warp
        self.tgt2aft_warp = tgt2aft_warp
        self.aft2tgt_warp = aft2tgt_warp
        self.pre2tgt_warp = pre2tgt_warp
        self.tgt2pre_mask = tgt2pre_mask
        self.tgt2aft_mask = tgt2aft_mask
        self.aft2tgt_mask = aft2tgt_mask
        self.pre2tgt_mask = pre2tgt_mask
        self.tgt2pre_flow = tgt2pre_flow
        self.tgt2aft_flow = tgt2aft_flow
        self.aft2tgt_flow = aft2tgt_flow
        self.pre2tgt_flow = pre2tgt_flow
        self.aft2tgt_error = aft2tgt_error
        self.pre2tgt_error = pre2tgt_error
        # self.tgt2aft_coords = tgt2aft_coords
        # self.aft2tgt_coords = aft2tgt_coords
        # self.pre2tgt_coords = pre2tgt_coords
        self.pred_depth_pre = pred_depth_pre
        self.pred_depth_tgt = pred_depth_tgt
        self.pred_depth_aft = pred_depth_aft
        self.pred_disp_pre = pred_disp_pre
        self.pred_disp_tgt = pred_disp_tgt
        self.pred_disp_aft = pred_disp_aft
        self.intrinsics = intrinsics
        self.tgt_image_f = tgt_image_f

    def build_flow_loss(self, res_flow_pre, res_flow_aft, pre_error, aft_error):
        ground_truth_pre = tf.stop_gradient(self.pre2tgt_flow + res_flow_pre)
        ground_truth_aft = tf.stop_gradient(self.aft2tgt_flow + res_flow_aft)
        pre_weight = 1.0 / (1.0 + tf.exp(5.0 * tf.reduce_mean(pre_error, axis=-1, keep_dims=True)))
        aft_weight = 1.0 / (1.0 + tf.exp(5.0 * tf.reduce_mean(aft_error, axis=-1, keep_dims=True)))
        pre_weight = tf.stop_gradient(pre_weight)
        aft_weight = tf.stop_gradient(aft_weight)
        self.flow_loss = tf.reduce_mean(tf.abs(ground_truth_pre - self.pre2tgt_flow) * tf.stop_gradient(pre_weight) + tf.abs(ground_truth_aft - self.aft2tgt_flow) * tf.stop_gradient(aft_weight))

    def get_gan_pair(self, step = 0.05, scale = 0.75):
        # generate random movement
        opt = self.opt
        random_xy = tf.truncated_normal([opt.batch_size, 2], mean = 0, stddev = step / 2, dtype = tf.float32, seed = None)
        random_z = tf.truncated_normal([opt.batch_size, 1], mean = -step, stddev = step / 2, dtype = tf.float32, seed = None)
        fixed_r = tf.zeros([opt.batch_size, 3])
        random_pose = tf.concat([random_xy, random_z, fixed_r], axis=-1)
        fake_flow = compute_rigid_flow(self.pred_depth_tgt[:,:,:,0], random_pose, self.intrinsics, False)
        fake_warp, fake_mask, fake_coords = forward_flow_warp(self.tgt_image_f, fake_flow, self.pred_depth_tgt)
        
        fake_tgt = fake_warp / fake_mask
        real_tgt = self.tgt_image_f * (fake_mask - 0.00001) / fake_mask

        _, h, w, _ = fake_tgt.get_shape().as_list()
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        offset_h = random.randint(0, int(h * (1 - scale)))
        offset_w = random.randint(0, int(h * (1 - scale)))

        return real_tgt[:, offset_h: offset_h + crop_h, offset_w: offset_w + crop_w, :], \
                fake_tgt[:, offset_h: offset_h + crop_h, offset_w: offset_w + crop_w, :]
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING_RESNET50 = 5
DISP_SCALING_VGG = 10
FLOW_SCALING = 0.1

def color_network(image, mask, vscope='cnet_d', reuse_weights=tf.AUTO_REUSE):
    isTrainable = opt.train_color_net
    return build_resnet50(tf.concat([image, mask], axis=-1), get_res_resnet50, isTrainable, vscope)

def disp_net(opt, dispnet_inputs):
    is_training = opt.train_disp_net
    return build_resnet50(dispnet_inputs, get_disp_resnet50, is_training, 'depth_net')

def flow_net(opt, flownet_inputs):
    is_training = (opt.add_flow_loss and opt.train_flow_net)
    return build_resnet50(flownet_inputs, get_flow, is_training, 'flow_net')

def pose_net(opt, posenet_inputs):
    is_training = opt.mode == 'train'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('pose_net') as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1  = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2  = slim.conv2d(conv1, 32,  5, 2)
            conv3  = slim.conv2d(conv2, 64,  3, 2)
            conv4  = slim.conv2d(conv3, 128, 3, 2)
            conv5  = slim.conv2d(conv4, 256, 3, 2)
            conv6  = slim.conv2d(conv5, 256, 3, 2)
            conv7  = slim.conv2d(conv6, 256, 3, 2)
            pose_pred = slim.conv2d(conv7, 12, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = 0.01 * tf.reshape(pose_avg, [-1, 2, 6])
            return pose_final

def critic(opt, image):
    is_training = opt.mode == 'train'
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as sc:
        with slim.arg_scope([slim.conv2d],
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params,
                    weights_regularizer=slim.l2_regularizer(0.0001),
                    activation_fn=tf.nn.leaky_relu):
            size = 64
            conv1  = slim.conv2d(image, size    ,  3, 2)
            conv2  = slim.conv2d(conv1, size * 2,  3, 2)
            conv3  = slim.conv2d(conv2, size * 4,  3, 2)
            conv4  = slim.conv2d(conv3, size * 8,  3, 2)
            ret    = slim.fully_connected(conv3, 1, activation_fn=None)
            return ret

def build_resnet50(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = conv(inputs, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            
            # DECODING
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            upconv6 = resize_like(upconv6, skip5)
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            upconv5 = resize_like(upconv5, skip4)
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            upconv4 = resize_like(upconv4, skip3)
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            pred4 = get_pred(iconv4)
            upred4  = upsample_nn(pred4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, upred4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            pred3 = get_pred(iconv3)
            upred3  = upsample_nn(pred3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, upred3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            pred2 = get_pred(iconv2)
            upred2  = upsample_nn(pred2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, upred2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            pred1 = get_pred(iconv1)

            return pred1

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, normalizer_fn=normalizer_fn)

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def get_disp_vgg(x):
    disp = DISP_SCALING_VGG * slim.conv2d(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_disp_resnet50(x):
    disp = DISP_SCALING_RESNET50 * conv(x, 1, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01
    return disp

def get_res_resnet50(x):
    return DISP_SCALING_RESNET50 * conv(x, 3, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.01

def get_flow(x):
    # Output flow value is normalized by image height/width
    flow = FLOW_SCALING * slim.conv2d(x, 2, 3, 1, activation_fn=None, normalizer_fn=None)
    return flow

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def upconv(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    cnv = conv(upsample, num_out_layers, kernel_size, 1)
    return cnv

def resconv(x, num_layers, stride):
    # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
    # but we preserve it here for consistency with Godard's implementation.
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv(x,         num_layers, 1, 1)
    conv2 = conv(conv1,     num_layers, 3, stride)
    conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.relu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
    out = x
    for i in range(num_blocks - 1):
        out = resconv(out, num_layers, 1)
    out = resconv(out, num_layers, 2)
    return out

class depth_predictor(object):
    def __init__(self, opt=None):
        self.opt = opt
        
    def build_model(self, pre_image_f, tgt_image_f, aft_image_f, intrinsics):
        opt = self.opt

        # pred_disp_pre = disp_net(opt, pre_image_f)
        # pred_disp_tgt = disp_net(opt, tgt_image_f)
        # pred_disp_aft = disp_net(opt, aft_image_f)

        pred_disp_pre = spatial_normalize(disp_net(opt, pre_image_f))
        pred_disp_tgt = spatial_normalize(disp_net(opt, tgt_image_f))
        pred_disp_aft = spatial_normalize(disp_net(opt, aft_image_f))

        pred_depth_pre = 1. / pred_disp_pre
        pred_depth_tgt = 1. / pred_disp_tgt
        pred_depth_aft = 1. / pred_disp_aft

        if intrinsics == None:
            self.pred_depth_tgt = pred_depth_tgt
            return

        pred_poses = pose_net(opt, tf.concat([tgt_image_f, pre_image_f, aft_image_f], axis=3))

        tgt2pre_flow = compute_rigid_flow(pred_depth_tgt[:,:,:,0], pred_poses[:,0,:], intrinsics, False)
        tgt2aft_flow = compute_rigid_flow(pred_depth_tgt[:,:,:,0], pred_poses[:,1,:], intrinsics, False)
        pre2tgt_flow = compute_rigid_flow(pred_depth_pre[:,:,:,0], pred_poses[:,0,:], intrinsics, True)
        aft2tgt_flow = compute_rigid_flow(pred_depth_aft[:,:,:,0], pred_poses[:,1,:], intrinsics, True)

        tgt2pre_mask = tf.ones_like(pre_image_f)
        tgt2aft_mask = tf.ones_like(aft_image_f)
        aft2tgt_mask = tf.ones_like(tgt_image_f)
        pre2tgt_mask = tf.ones_like(tgt_image_f)

        if opt.use_forward_warp:
            # forward 
            tgt2pre_warp, tgt2pre_mask, tgt2pre_coords = forward_flow_warp(tgt_image_f, tgt2pre_flow, pred_depth_tgt)
            tgt2aft_warp, tgt2aft_mask, tgt2aft_coords = forward_flow_warp(tgt_image_f, tgt2aft_flow, pred_depth_tgt)
            aft2tgt_warp, aft2tgt_mask, aft2tgt_coords = forward_flow_warp(aft_image_f, aft2tgt_flow, pred_depth_aft)
            pre2tgt_warp, pre2tgt_mask, pre2tgt_coords = forward_flow_warp(pre_image_f, pre2tgt_flow, pred_depth_pre)

        else:
            # backward
            tgt2pre_warp = flow_warp(tgt_image_f, pre2tgt_flow)
            tgt2aft_warp = flow_warp(tgt_image_f, aft2tgt_flow)
            aft2tgt_warp = flow_warp(aft_image_f, tgt2aft_flow)
            pre2tgt_warp = flow_warp(pre_image_f, tgt2pre_flow)

            tgt2pre_mask = pred_depth_pre
            tgt2aft_mask = pred_depth_aft
            aft2tgt_mask = pred_depth_tgt
            pre2tgt_mask = pred_depth_tgt

        if opt.use_color_net:
            tgt2pre_warp = color_network(tgt2pre_warp, tgt2pre_mask)
            tgt2aft_warp = color_network(tgt2aft_warp, tgt2aft_mask)
            aft2tgt_warp = color_network(aft2tgt_warp, aft2tgt_mask)
            pre2tgt_warp = color_network(pre2tgt_warp, pre2tgt_mask)

        aft2tgt_error = image_similarity(aft2tgt_warp, tgt_image_f * aft2tgt_mask)
        pre2tgt_error = image_similarity(pre2tgt_warp, tgt_image_f * pre2tgt_mask)
        tgt2aft_error = image_similarity(tgt2aft_warp, aft_image_f * tgt2aft_mask)
        tgt2pre_error = image_similarity(tgt2pre_warp, pre_image_f * tgt2pre_mask)

        tgt_error = tf.reduce_mean(aft2tgt_error)
        tgt_error +=tf.reduce_mean(pre2tgt_error)
        aft_error = tf.reduce_mean(image_similarity(tgt2aft_warp, aft_image_f))
        pre_error = tf.reduce_mean(image_similarity(tgt2pre_warp, pre_image_f))

        # tgt_error = tf.reduce_sum(image_similarity(aft2tgt_warp, tgt_image_f * aft2tgt_mask)) / tf.reduce_sum(aft2tgt_mask)
        # tgt_error +=tf.reduce_sum(image_similarity(pre2tgt_warp, tgt_image_f * pre2tgt_mask)) / tf.reduce_sum(pre2tgt_mask)
        # aft_error = tf.reduce_sum(image_similarity(tgt2aft_warp, aft_image_f * tgt2aft_mask)) / tf.reduce_sum(tgt2aft_mask)
        # pre_error = tf.reduce_sum(image_similarity(tgt2pre_warp, pre_image_f * tgt2pre_mask)) / tf.reduce_sum(tgt2pre_mask)

        tgt_loss = tgt_error
        src_loss = aft_error + pre_error
        smooth_loss = 0.1 * (compute_smooth_loss(pred_disp_pre, pre_image_f) + compute_smooth_loss(pred_disp_aft, aft_image_f) + compute_smooth_loss(pred_disp_tgt, tgt_image_f))
        loss = tf.reduce_mean(tgt2pre_error) + tf.reduce_mean(aft2tgt_error)
        loss = tf.reduce_mean(pre2tgt_error) + tf.reduce_mean(tgt2aft_error)

        self.loss = loss
        self.tgt_loss = tgt_loss
        self.src_loss = src_loss
        self.smooth_loss = smooth_loss
        self.tgt2pre_warp = tgt2pre_warp
        self.tgt2aft_warp = tgt2aft_warp
        self.aft2tgt_warp = aft2tgt_warp
        self.pre2tgt_warp = pre2tgt_warp
        self.tgt2pre_mask = tgt2pre_mask
        self.tgt2aft_mask = tgt2aft_mask
        self.aft2tgt_mask = aft2tgt_mask
        self.pre2tgt_mask = pre2tgt_mask
        self.tgt2pre_flow = tgt2pre_flow
        self.tgt2aft_flow = tgt2aft_flow
        self.aft2tgt_flow = aft2tgt_flow
        self.pre2tgt_flow = pre2tgt_flow
        self.aft2tgt_error = aft2tgt_error
        self.pre2tgt_error = pre2tgt_error
        # self.tgt2aft_coords = tgt2aft_coords
        # self.aft2tgt_coords = aft2tgt_coords
        # self.pre2tgt_coords = pre2tgt_coords
        self.pred_depth_pre = pred_depth_pre
        self.pred_depth_tgt = pred_depth_tgt
        self.pred_depth_aft = pred_depth_aft
        self.pred_disp_pre = pred_disp_pre
        self.pred_disp_tgt = pred_disp_tgt
        self.pred_disp_aft = pred_disp_aft
        self.intrinsics = intrinsics
        self.tgt_image_f = tgt_image_f

    def build_flow_loss(self, res_flow_pre, res_flow_aft, pre_error, aft_error):
        ground_truth_pre = tf.stop_gradient(self.pre2tgt_flow + res_flow_pre)
        ground_truth_aft = tf.stop_gradient(self.aft2tgt_flow + res_flow_aft)
        pre_weight = 1.0 / (1.0 + tf.exp(5.0 * tf.reduce_mean(pre_error, axis=-1, keep_dims=True)))
        aft_weight = 1.0 / (1.0 + tf.exp(5.0 * tf.reduce_mean(aft_error, axis=-1, keep_dims=True)))
        pre_weight = tf.stop_gradient(pre_weight)
        aft_weight = tf.stop_gradient(aft_weight)
        self.flow_loss = tf.reduce_mean(tf.abs(ground_truth_pre - self.pre2tgt_flow) * tf.stop_gradient(pre_weight) + tf.abs(ground_truth_aft - self.aft2tgt_flow) * tf.stop_gradient(aft_weight))

    def get_gan_pair(self, step = 0.05, scale = 0.75):
        # generate random movement
        opt = self.opt
        random_xy = tf.truncated_normal([opt.batch_size, 2], mean = 0, stddev = step / 2, dtype = tf.float32, seed = None)
        random_z = tf.truncated_normal([opt.batch_size, 1], mean = -step, stddev = step / 2, dtype = tf.float32, seed = None)
        fixed_r = tf.zeros([opt.batch_size, 3])
        random_pose = tf.concat([random_xy, random_z, fixed_r], axis=-1)
        fake_flow = compute_rigid_flow(self.pred_depth_tgt[:,:,:,0], random_pose, self.intrinsics, False)
        fake_warp, fake_mask, fake_coords = forward_flow_warp(self.tgt_image_f, fake_flow, self.pred_depth_tgt)
        
        fake_tgt = fake_warp / fake_mask
        real_tgt = self.tgt_image_f * (fake_mask - 0.00001) / fake_mask

        _, h, w, _ = fake_tgt.get_shape().as_list()
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        offset_h = random.randint(0, int(h * (1 - scale)))
        offset_w = random.randint(0, int(h * (1 - scale)))

        return real_tgt[:, offset_h: offset_h + crop_h, offset_w: offset_w + crop_w, :], \
                fake_tgt[:, offset_h: offset_h + crop_h, offset_w: offset_w + crop_w, :]

sess = None
input_float32 = None
model = None

flags = tf.app.flags
flags.DEFINE_string("mode",                         "",    "train or test")
flags.DEFINE_string("name",                         "",    "the name for a train")
flags.DEFINE_string("dataset_dir",                  "",    "Dataset directory")
flags.DEFINE_string("init_ckpt_file",             None,    "Specific checkpoint file to initialize from")
flags.DEFINE_integer("batch_size",                   1,    "The size of of a sample batch")
flags.DEFINE_integer("img_height",                 128,    "Image height")
flags.DEFINE_integer("img_width",                  416,    "Image width")
flags.DEFINE_string("checkpoint_dir",               "",    "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate",             0.0002,    "Learning rate for adam")
flags.DEFINE_string("output_dir",                 None,    "Test result output directory")
flags.DEFINE_integer("max_to_keep",                 20,    "Maximum number of checkpoints to save")
flags.DEFINE_integer("max_steps",             30000000,    "Maximum number of training iterations")
flags.DEFINE_integer("save_ckpt_freq",            5000,    "Save the checkpoint model every save_ckpt_freq iterations")
flags.DEFINE_integer("print_freq",                 100,    "print loss freq")
flags.DEFINE_integer("num_threads",                  1,    "Number of threads for data loading")
flags.DEFINE_boolean("shuffle_batch",             True,    "shuffle batch or not")
flags.DEFINE_boolean("one_batch",                False,    "train one batch and stop")
flags.DEFINE_boolean("use_color_net",            False,    "use color net or not")
flags.DEFINE_boolean("train_color_net",           True,    "train color net or not")
flags.DEFINE_boolean("train_disp_net",            True,    "train disp net or not")
flags.DEFINE_boolean("add_flow_loss",            False,    "add flow loss or not")
flags.DEFINE_boolean("train_flow_net",           False,    "train flow net or use pre trained flow net")
flags.DEFINE_boolean("use_forward_warp",          True,    "true for forward wrap and false for backward warp")
flags.DEFINE_boolean("add_discriminator",        False,    "add discriminator or not")


opt = flags.FLAGS

def setup(ckpt):
    global sess
    global input_float32
    global opt
    global model
    ##### init #####
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    input_float32 = tf.placeholder(tf.float32, [1,
                128, 416, 3], name='raw_input')
    model = depth_predictor(opt)
    model.build_model(input_float32, input_float32, input_float32, None)

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ##### Go #####
    sess = tf.Session(config=config)
    saver.restore(sess, ckpt)

def inference(image):
    global sess
    global input_float32
    global model
    size = image.shape
    image = pil.fromarray(image)
    scaled_im = image.resize((416, 128), pil.ANTIALIAS)
    inputs = np.expand_dims(scaled_im, 0)
    inputs = (inputs / 128.0) - 1.0
    fetches = { "depth": model.pred_depth_tgt }
    pred = sess.run(fetches, feed_dict = {input_float32: inputs})
    res = pred["depth"][0][:,:,0]
    res = pil.fromarray(res)
    res = res.resize((size[1], size[0]))
    res = np.array(res)
    res = 1. / res
    return (res - np.min(res)) * 255 / (np.max(res) - np.min(res))

# test_depth.setup("/home/wuxiaoshi/pre_trained/dispnet/model_sn")
# x = PIL.Image.open("/home/wuxiaoshi/GeoNet-master_original/abs_sum/src.png")
# ouput = test_depth.inference(x)
