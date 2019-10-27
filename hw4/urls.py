"""frontend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url
from . import view
from . import rules

rules.initialize_db()

urlpatterns = [
    url(r'^index$', view.show_index),
    url(r'^logon$', view.show_logon),
    url(r'^login$', view.show_login),
    url(r'^error$', view.show_error),
    url(r'^logout$', view.show_logout),
    url(r'^service/$', view.show_service),
    url(r'^history/[0-9]*$', view.show_history),
    url(r'^detail/[0-9a-zA-Z_]+/[0-9]+$', view.show_details),
    url(r'^delete/[0-9a-zA-Z_]+/[0-9]+$',view.show_delete),
    url(r'^not_found$', view.show_not_found),
    url(r'^delete_many$', view.delete_many)

]
