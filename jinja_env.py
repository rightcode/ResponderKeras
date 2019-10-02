"""
jinja_env.py
Jinja2のフィルタを定義する
"""


def static_filter(path):
    return '/static/' + path


def image_filter(path):
    return '/static/images/' + path


def css_filter(path):
    return '/static/css/' + path


def script_filter(path):
    return '/static/script/' + path


def badge_filter(text):
    return '<span class="badge badge-secondary">' + text + '</span>'


def badge_active_filter(text):
    return '<span class="badge badge-primary">' + text + '</span>'


def fc_filter(neurons):
    """
    全結合中間層の追加フィルタ
    :param neurons:
    :return:
    """
    neurons = str(neurons)
    return '<button class="btn btn-secondary btn-sm" onclick="add_fc(' + neurons + ')">' + neurons + ' neurons</button>'

