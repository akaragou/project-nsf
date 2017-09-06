def python_postgresql():
    connect_str = "dbname='mircs' user='mircs' host='localhost' " + \
              "password='serrelab'"
    return connect_str


def credentials():
    return 'pclpslabserrecit3.services.brown.edu', 22, 'youssef', 'FuzzyBlackViper49'


def postgresql_credentials():
    return {
            'username': 'mircs',
            'password': 'serrelab'
           }


def postgresql_connection(port=''):
    unpw = postgresql_credentials()
    params = {
        'database': 'mircs',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params


def g15_credentials():
    return {
            'username': 'clickme',
            'password': 'do not click'
           }


def results_postgresql_credentials():
    return {
            'username': 'clickme',
            'password': 'serrelab'
           }


def x9_postgresql_connection():
    unpw = results_postgresql_credentials()
    params = {
        'database': 'clickme',
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost'
    }
    return params
