"""
urls.py
ここではURLとコントローラのルーティング、およびJinja2のフィルタ設定を行う。
"""

import jinja_env
from controllers import *

api.jinja_env.filters.update(
    static=jinja_env.static_filter,
    image=jinja_env.image_filter,
    css=jinja_env.css_filter,
    script=jinja_env.script_filter,
    badge=jinja_env.badge_filter,
    badge_active=jinja_env.badge_active_filter,
    fc=jinja_env.fc_filter,
)

# ルーティング
api.add_route('/', IndexController)
api.add_route('/create/{dataset}', CreateController)
api.add_route('/learn/{dataset}', LearnController)
api.add_route('/learning/{uid}', LearningController)
api.add_route('/404', NotFoundController)

