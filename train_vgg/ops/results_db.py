#!/usr/bin/env python
import os
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
from hyperparameter_optimization_parameters import create_combos

db_schema_file = os.path.join('ops', 'db_schema.txt')


def csv_list(attention_layers):
    return ','.join(attention_layers)


class db(object):
    def __init__(self, config):
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        pgsql_string = credentials.x9_postgresql_connection()
        self.pgsql_string = pgsql_string
        self.conn = psycopg2.connect(**pgsql_string)
        self.conn.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.cur = self.conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_db()
        if exc_type is not None:
            print exc_type, exc_value, traceback
        return self

    def close_db(self):
        self.cur.close()
        self.conn.close()

    def init_db(self):
        db_schema = open(db_schema_file).read().splitlines()
        for s in db_schema:
            t = s.strip()
            if len(t):
                self.cur.execute(t)
        self.conn.commit()

    def create_db_combo(self, dct):
        self.cur.execute(
            """
            INSERT INTO combos
            (fine_tune_layers, new_lr, data_augmentations)
            values ('%s', %s, '%s') RETURNING _id
            """
            % (csv_list(dct['fine_tune_layers']), dct['new_lr'][0],
                csv_list(dct['data_augmentations'])))
        self.conn.commit()
        return self.cur.fetchone()['_id']

    def append_results(self, data):
        self.cur.execute(
            """
            INSERT INTO results
            (combos_table_id, model_name, iteration, loss, validation_accuracy)
            values (%s, '%s', %s, %s, %s)
            """
            % (data['combos_table_id'], data['model_name'],
                data['iteration'], data['loss'], data['validation_accuracy']))
        self.conn.commit()

    def get_configuration(self):
        self.cur.execute(
            """
            SELECT * from combos
            WHERE processing is NULL LIMIT 1
            """)
        row = self.cur.fetchone()
        if row is not None:
            row = dict(row)
            self.cur.execute(
                """
                UPDATE combos
                SET processing=True
                where _id=%s
                """ % row['_id'])
        # Fix for how we have to store attention_layers
        row['attention_layers'] = row['attention_layers'].split(',')
        return row


def initialize_database():
    parameter_combos = create_combos()
    creds = credentials.results_postgresql_credentials()
    with db(creds) as db_conn:
        db_conn.init_db()
        for comb in parameter_combos:
            db_conn.create_db_combo(comb)
    print 'Initialized database tables and combos'


if __name__ == '__main__':
    initialize_database()
