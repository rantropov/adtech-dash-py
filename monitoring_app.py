import sqlite3
from flask import Flask, g, jsonify, request
import os

import classifier

# create our little application :)
app = Flask(__name__)

# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'monitoring_app.db'),
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
app.config.from_envvar('MONITORING_APP_SETTINGS', silent=True)


def connect_db():
    """Connects to the specific database"""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_db()
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Closes the database again at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initializes the database."""
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


@app.route('/received_ads, methods=['POST'])
def post_received_ad_event(value):
    """Puts ad value with current time into database"""
    db = get_db()

    # Classify ad

    # Insert into database
    db.execute(
        '''

        ''',
        [value])
    db.commit()
    return jsonify(value=value)


@app.route('/fscore')
def get_fscore():
    """Gets fscore from database"""
    pass

if __name__ == '__main__':
    app.run()