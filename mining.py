import cPickle as pickle
from operator import attrgetter
import time
import datetime
import functools
import copy

import sqlite3
from dateutil.relativedelta import relativedelta
import scb.names

import settings


class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)


def db_type(k):
    return {
        int: 'INTEGER', 
        float: 'FLOAT',
        }.get(k)


WORKER = 'worker'
ONCALL = 'oncall'
SIGNUP = 'signup'
BUYER = 'buyer'
CUSTOMER = 'customer'
ALL = 'all'
MALE = 'male'
FEMALE = 'female'
UNKNOWN = 'unknown'
GENDERS = [MALE, FEMALE, UNKNOWN]
CLASSES = [WORKER, ONCALL, SIGNUP, BUYER, CUSTOMER] + GENDERS

CUP_LITERS = 0.237


@memoized
def adapt_datetype(ts):
    return int(time.mktime(ts.timetuple()))
@memoized
def convert_datetype(klass, s):
    return klass.fromtimestamp(float(s))
for klass, prefix in [
        (datetime.datetime, 'datetime'),
        (datetime.date, 'date')]:
    convert = functools.partial(convert_datetype, klass)
    sqlite3.register_adapter(klass, adapt_datetype)
    sqlite3.register_converter('py%s' % prefix, convert)

conn = sqlite3.connect('mining.sqlite', detect_types=sqlite3.PARSE_DECLTYPES)
conn.row_factory = sqlite3.Row
def get_cursor():
    return conn.cursor()
c = get_cursor()

import_schema = [
    "DROP TABLE IF EXISTS sections;",
    """CREATE TABLE sections (
        name CHARACTER VARYING(20) PRIMARY KEY
        )""",
    "DROP TABLE IF EXISTS classes;",
    """CREATE TABLE classes (
        id CHARACTER VARYING(20) PRIMARY KEY,
        name CHARACTER VARYING(20)
        )""",
    "DROP TABLE IF EXISTS users;",
    """CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name CHARACTER VARYING(50),
        gender CHARACTER VARYING(20) DEFAULT "u",
        section CHARACTER VARYING(20) DEFAULT ""
        )""",
    "DROP TABLE IF EXISTS buys;",
    """CREATE TABLE buys (
        id INTEGER PRIMARY KEY,
        put_at pydatetime,
        user INTEGER REFERENCES users
        )""",
    "DROP TABLE IF EXISTS semesters;",
    """CREATE TABLE semesters (
        id INTEGER PRIMARY KEY,
        name CHARACTER(6),
        start pydate,
        end pydate
        )""",
    "DROP TABLE IF EXISTS shifts;",
    """CREATE TABLE shifts (
        id INTEGER PRIMARY KEY,
        date pydate,
        span TYNYINT,
        semester INTEGER REFERENCES semesters
        )""",
    "DROP TABLE IF EXISTS signups;",
    """CREATE TABLE signups (
        id INTEGER PRIMARY KEY,
        shift INTEGER REFERENCES shifts,
        user INTEGER REFERENCES user,
        klass CHARACTER VARYING(20) REFERENCES classes
        )""",
    """CREATE INDEX buy_user_idx ON buys(user)""",
    """CREATE INDEX shift_semester_idx ON shifts(semester)""",
    """CREATE INDEX signup_shift_idx ON signups(shift)""",
    """CREATE INDEX signup_user_idx ON signups(user)""",
    """CREATE INDEX signup_klass_idx ON signups(klass)""",
    ]
for g in CLASSES:
    import_schema.append(
        """INSERT INTO classes (id, name) VALUES ("%s", "%s");""" % (g, g))

def do_queries(queries):
    for query in queries:
        try:
            c.execute(query)
        except sqlite3.OperationalError:
            print query
            raise
    conn.commit()


class HashError(Exception):
    pass


class Hash(object):

    def __init__(self, row):
        if row is None:
            raise HashError(row)
        self.cols = []
        self.values = []
        for idx, col in enumerate(row.keys()):
            self.cols.append(col)
            self.values.append(row[idx])
            setattr(self, col, row[idx])
        self.items = zip(self.cols, self.values)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def __hash__(self):
        return hash(self.id)

    def __int__(self):
        return self.id


def hashes(res):
    return [Hash(r) for r in res]


@memoized
def get_interval(interval):
    c.execute(
        "SELECT * FROM intervals WHERE start = ? AND end = ? LIMIT 1",
        (interval,))
    return Hash(c.fetchone())


@memoized
def get_intervals():
    c.execute("SELECT * FROM intervals ORDER BY start")
    return hashes(c.fetchall())


@memoized
def buys_within(interval):
    c.execute(
        "SELECT * FROM interval_buys WHERE interval = ?;",
        (interval.id,))
    return hashes(c.fetchall())


@memoized
def user_buys_within(interval, user):
    c.execute(
        """SELECT count(*) as count FROM interval_buys
            WHERE interval = ? AND user = ?;""",
            (interval.id, user.id))
    return Hash(c.fetchone()).count


@memoized
def shifts_within(interval):
    c.execute(
        "SELECT * FROM interval_shifts WHERE interval = ?;""",
        (interval.id,))
    return hashes(c.fetchall())


@memoized
def signups_within(interval, g=None):
    extra = ""
    args = [interval.id,]
    if g in [WORKER, ONCALL]:
        extra = ' AND klass = ?',
        args.append(g)
    c.execute(
        """SELECT * FROM interval_signups WHERE interval = ?%s;""" % extra,
        args)
    return hashes(c.fetchall())


@memoized
def get_users(iterable, get_set=False):
    user_ids = set()
    for i in iterable:
        if isinstance(i, int):
            user_ids.add(str(i))
        else:
            user_ids.add(str(i.user))
    c.execute(
        "SELECT * FROM users WHERE id IN (%s);" % ", ".join(user_ids))
    return hashes(c.fetchall())


@memoized
def with_gender(gender, users):
    assert gender in GENDERS
    return filter(lambda user: user.gender == gender, users)


@memoized
def class_within(interval, g):
    if g in [WORKER, ONCALL]:
        return get_users(signups_within(interval, g))
    elif g == SIGNUP:
        return get_users(signups_within(interval))
    elif g == BUYER:
        return get_users(buys_within(interval))
    elif g == CUSTOMER:
        getter = attrgetter('user')
        def get_set(func):
            return set(map(getter, func(interval)))
        buyers = get_set(buys_within)
        signups = get_set(signups_within)
        customers = buyers.difference(signups)
        return get_users(customers)
    elif g == ALL:
        iterable = []
        iterable.extend(signups_within(interval))
        iterable.extend(buys_within(interval))
        return get_users(iterable)
    elif g in GENDERS:
        return with_gender(g, class_within(interval, ALL))
    assert False, g


@memoized
def classes_within(interval):
    classes = {}
    for g in AGG_CLASSES:
        classes[g] = class_within(interval, g)
    return classes


@memoized
def traverse(obj, func, test=None, context=None):
    if context is not None:
        context = copy.copy(context)
    if isinstance(obj, dict):
        for key, val in obj.items():
            if not callable(test) or test(key):
                obj[key] = traverse(val, func, test, context)
        return obj
    return func(obj)


@memoized
def _gender_node(obj):
    new = {}
    for gender in GENDERS:
        func = functools.partial(with_gender, gender)
        new[gender] = func(obj)
    new[ALL] = obj
    return new
def _gender_test(key):
    return key not in [ALL] + GENDERS
def get_gendered(interval, obj):
    return traverse(obj, _gender_node, _gender_test)


@memoized
def _interval_buy_node(interval, users):
    return [user_buys_within(interval, user) for user in users]
def get_buys(interval, obj):
    func = functools.partial(_interval_buy_node, interval)
    return traverse(obj, func)


def get_counted(interval, obj):
    return traverse(obj, len)


def get_summed(interval, obj):
    return traverse(obj, sum)


def insert_interval(name, start, end):
    c.execute(
        "SELECT * FROM buys WHERE put_at >= ? AND put_at < ?",
        (start, end + relativedelta(days=1)))
    buys = hashes(c.fetchall())

    c.execute(
        "SELECT * FROM shifts WHERE date >= ? AND date <= ?",
        (start, end))
    shifts = hashes(c.fetchall())
    c.execute("SELECT count(*) AS offset FROM interval_shifts;")

    c.execute(
        "SELECT * FROM signups WHERE shift in (%s);" % \
            ", ".join([str(shift.id) for shift in shifts]))
    signups = hashes(c.fetchall())

    c.execute(
        """INSERT INTO intervals 
            (name, start, end, buy_count)
            VALUES (?, ?, ?, ?);""",
        (name, start, end, len(buys)))
    idx = c.lastrowid

    c.executemany(
        "INSERT INTO interval_buys (interval, user, put_at) VALUES (?, ?, ?);",
        [(idx, buy.user, buy.put_at) for buy in buys])
    c.executemany(
        """INSERT INTO interval_shifts
            (id, interval, date, span, semester) VALUES (?, ?, ?, ?, ?);""",
        [(shift.id, idx, shift.date, shift.span, shift.semester) for shift in shifts])

    c.executemany(
        """INSERT INTO interval_signups
            (interval, shift, user, klass) VALUES (?, ?, ?, ?);""",
        [(idx, signup.shift, signup.user, signup.klass) for signup in signups])


extra_schema = [
    "DROP TABLE IF EXISTS intervals;",
    """CREATE TABLE intervals (
        id INTEGER PRIMARY KEY,
        name CHARACTER VARYING(50) DEFAULT "",
        start pydate,
        end pydate,
        buy_count INTEGER
        )""",
    "DROP TABLE IF EXISTS interval_shifts;",
    """CREATE TABLE interval_shifts (
        id INTEGER PRIMARY KEY,
        interval INTEGER REFERENCES intervals,
        date pydate,
        span TYNYINT,
        semester INTEGE REFERENCES semesters
        )""",
    "DROP TABLE IF EXISTS interval_signups;",
    """CREATE TABLE interval_signups (
        id INTEGER PRIMARY KEY,
        interval INTEGER REFERENCES intervals,
        shift INTEGER REFERENCES interval_shifts,
        user INTEGER REFERENCES users,
        klass CHARACTER VARYING(20) REFERENCES classes
        )""",
    "DROP TABLE IF EXISTS interval_buys;",
    """CREATE TABLE interval_buys (
        id INTEGER PRIMARY KEY,
        interval INTEGER REFERENCES intervals,
        user INTEGER REFERENCES users,
        put_at pydatetime
        )""",
    "DROP TABLE IF EXISTS interval_buy_counts;",
    """CREATE TABLE interval_buy_counts (
        id INTEGER PRIMARY KEY,
        interval INTEGER REFERENCES intervals,
        user INTEGER REFERENCES users,
        count INTEGER
        )""",
    """CREATE INDEX i_shift_interval_idx ON interval_shifts(interval)""",
    """CREATE INDEX i_shift_semester_idx ON interval_shifts(semester)""",
    """CREATE INDEX i_signup_interval_idx ON interval_signups(interval)""",
    """CREATE INDEX i_signup_shift_idx ON interval_signups(shift)""",
    """CREATE INDEX i_signup_user_idx ON interval_signups(user)""",
    """CREATE INDEX i_signup_klass_idx ON interval_signups(klass)""",
    """CREATE INDEX i_buy_interval_idx ON interval_buys(interval)""",
    """CREATE INDEX i_buy_user_idx ON interval_buys(user)""",
    """CREATE INDEX i_buy_count_interval_idx ON interval_buy_counts(interval)""",
    """CREATE INDEX i_buy_count_user_idx ON interval_buy_counts(user)""",
    ]


def transform(interval, base, steps):
    if steps:
        return transform(interval, steps[0](interval, base), steps[1:])
    return base


@memoized
def get_metric(interval, metric, klass, gender):
    c.execute(
        """SELECT * FROM %(db_plural)s WHERE 
            interval = ? AND 
            klass = ? AND
            gender = ? LIMIT 1""" % get_metric_context(metric),
        (interval.id, klass, gender))
    return Hash(c.fetchone())


def dictzip(*dicts):
    result = dict()
    dict_keys = []
    non_dict_keys = []
    for k, v in dicts[0].items():
        if isinstance(v, dict):
            dict_keys.append(k)
        else:
            non_dict_keys.append(k)
            result[k] = []
    for k in dict_keys:
        result[k] = dictzip(*[d[k] for d in dicts])
    for d in dicts:
        for k in non_dict_keys:
            result[k].append(d[k])
    return result


AGG_CLASSES = [CUSTOMER, WORKER]
METRICS = {
    'buys': {
        'py_type': int,
        'basic': True,
        'transforms': (get_buys, get_summed),
        },
    'people': {
        'py_type': int,
        'basic': True,
        'transforms': (get_counted,),
        },
    'buys_per_person': {
        'py_type': float,
        'basic': False,
        },
    'liters': {
        'py_type': float,
        'basic': False,
        },
    'liters_per_person': {
        'py_type': float,
        'basic': False,
        },
    }


def get_metrics(metric_type=False):
    metric_hashes = []
    for metric, meta in METRICS.items():
        if metric_type and metric != metric_type:
            continue
        context = get_metric_context(metric)
        c.execute("SELECT * FROM %(db_plural)s ORDER BY id" % context)
        metric_hashes.extend(hashes(c.fetchall()))
    return metric_hashes


def clear_metrics(metric_type=False):
    for metric, meta in METRICS.items():
        if metric_type and metric != metric_type:
            continue
        context = get_metric_context(metric)
        c.execute("DELETE FROM %(db_plural)s" % context)


def get_metric_context(metric):
    meta = METRICS[metric]
    return dict(
        db_singular='metric_%s' % metric,
        db_plural='metric_%ss' % metric,
        py_type=meta['py_type'],
        db_type=db_type(meta['py_type']),
        )


def get_metric_schema(metric):
    templates = [
        "DROP TABLE IF EXISTS %(db_plural)s;",
        """CREATE TABLE %(db_plural)s (
            id INTEGER PRIMARY KEY,
            interval INTEGER REFERENCES intervals,
            klass CHARACTER VARYING(20) REFERENCES classes,
            gender CHARACTER VARYING(20) DEFAULT "u",
            value %(db_type)s
            )""",
        """CREATE INDEX %(db_plural)s_interval_idx ON %(db_plural)s(interval)""",
        """CREATE INDEX %(db_plural)s_klass_idx ON %(db_plural)s(klass)""",
    ]
    return [template % get_metric_context(metric) for template in templates]


def insert_metric(interval, metric, klass, gender, value, commit=True):
    sql = """INSERT INTO %(db_plural)s
        (interval, klass, gender, value)
        VALUES (?, ?, ?, ?);""" % get_metric_context(metric)
    c.execute(sql, (int(interval), klass, gender, value))
    if commit:
        conn.commit()


def do_basic_metrics():
    basic_metrics = [metric for metric, meta in METRICS.items() if meta['basic']]
    for metric in basic_metrics:
        intervals = get_intervals()
        meta = METRICS[metric]
        for interval in intervals:
            gendered = get_gendered(interval, classes_within(interval))
            data = transform(interval, copy.deepcopy(gendered), meta['transforms'])
            for klass in AGG_CLASSES:
                for gender in [ALL] + GENDERS:
                    args = (
                        interval, 
                        metric, 
                        klass, 
                        gender, 
                        data[klass][gender],
                        )
                    insert_metric(*args, commit=False)
    conn.commit()


def do_extra_metrics():
    buys = get_metrics('buys')
    people = get_metrics('people')

    clear_metrics('buys_per_person')
    for b, p in zip(buys, people):
        mean = float(b.value) / max(1.0, p.value)
        insert_metric(
            b.interval, 
            'buys_per_person', 
            b.klass, 
            b.gender, 
            mean, commit=False,
            )
    buys_per_person = get_metrics('buys_per_person')

    clear_metrics('liters')
    for b in buys:
        insert_metric(
            b.interval, 
            'liters', 
            b.klass, 
            b.gender, 
            b.value * CUP_LITERS, commit=False,
            )

    clear_metrics('liters_per_person')
    for bpp in buys_per_person:
        insert_metric(
            bpp.interval, 
            'liters_per_person', 
            bpp.klass, 
            bpp.gender, 
            bpp.value * CUP_LITERS, commit=False,
            )

    conn.commit()


def num(obj):
    try:
        return len(obj)
    except TypeError:
        return int(obj)


def ratio(obj, pool):
    return float(num(obj)) / max(1.0, num(pool))


def formatpc(obj, pool):
    r = ratio(obj, pool)
    pad = 7
    if r > 1.0:
        return "#" * pad
    else:
        s = "%.2f%%" % (100.0 * r)
        return s.rjust(pad)


def get_semesters():
    c.execute("SELECT * FROM semesters ORDER BY start;")
    return hashes(c.fetchall())


def main():
    if settings.DO_IMPORT:
        do_queries(import_schema)

        name_gender = {}
        male_names = scb.names.male()
        female_names = scb.names.female()
        for names, gender in [(male_names, MALE), (female_names, FEMALE)]:
            for name in names:
                name_gender[name] = gender

        with open('mining.pkl', 'rb') as f:
            d = pickle.load(f)
        del f

        for sec in d['sections']:
            c.execute("INSERT INTO sections (name) VALUES (?)", (sec,))
        conn.commit()

        for uid in d['users']:
            name = d['user_name'][uid]
            gender = name_gender.get(name, UNKNOWN)
            section = d['user_section'][uid]
            c.execute(
                """INSERT INTO users 
                    (id, name, gender, section)
                    VALUES (?, ?, ?, ?)
                    """, 
                (uid, name, gender, section))
        conn.commit()

        buy_args = []
        for user, buys in d['user_orders'].items():
            buy_args.extend((put_at, user) for put_at in buys)
        c.executemany(
            """INSERT INTO buys 
                (put_at, user)
                VALUES (?, ?)
                """, 
            buy_args)
        conn.commit()

        for sem in d['semesters']:
            name = d['semester_name'][sem]
            start, end = d['semester_startend'][sem]
            c.execute(
                """INSERT INTO semesters 
                    (id, name, start, end)
                    VALUES (?, ?, ?, ?)
                    """, 
                (sem, name, start, end))
        conn.commit()

        for shift in d['shifts']:
            date = d['shift_date'][shift]
            span = d['shift_span'][shift]
            sem = d['shift_semester'][shift]
            c.execute(
                """INSERT INTO shifts 
                    (id, date, span, semester)
                    VALUES (?, ?, ?, ?)
                    """, 
                (shift, date, span, sem))
        conn.commit()

        signup_args = []
        for signup_key, signup_class in [
                ('user_workshifts', WORKER),
                ('user_oncallshifts', ONCALL)]:
            for user, shifts in d[signup_key].items():
                signup_args.extend([(shift, user, signup_class) for shift in shifts])
        c.executemany(
            """INSERT INTO signups
                (shift, user, klass)
                VALUES (?, ?, ?)
                """,
            signup_args)
        conn.commit()

    if settings.DO_EXTRAS:
        do_queries(extra_schema)

        for sem in get_semesters():
            insert_interval(sem.name, sem.start, sem.end)
        conn.commit()

    basic_metrics = [metric for metric, meta in METRICS.items() if meta['basic']]
    if settings.DO_BASIC_METRICS:
        for metric in basic_metrics:
            do_queries(get_metric_schema(metric))
        do_basic_metrics()
    extra_metrics = [metric for metric, meta in METRICS.items() if not meta['basic']]
    if settings.DO_EXTRA_METRICS:
        for metric in extra_metrics:
            do_queries(get_metric_schema(metric))
        do_extra_metrics()

    if settings.DO_CSV:
        for metric, meta in METRICS.items():
            csv_file = "%s.csv" % metric
            fields = ['interval']
            lookups = []
            for klass in AGG_CLASSES:
                for gender in [ALL, MALE, FEMALE]:
                    lookups.append((klass, gender))
                    fields.append("%s_%s" % (klass, gender))
            with open(csv_file, 'w') as csv:
                line = ", ".join(map(str, fields))
                csv.write(line + '\n')
            for interval in get_intervals():
                try:
                    values = []
                    for lookup in lookups:
                        values.append(get_metric(interval, metric, *lookup).value)
                    total = sum(values)
                    if total == 0:
                        continue
                    if metric == 'buys' and total < 500:
                        continue
                    if metric == 'people' and total < 250:
                        continue
                    line = ",".join(map(str, [interval.name] + values))
                    with open(csv_file, 'a+') as csv:
                        csv.write(line + '\n')
                except HashError:
                    pass



if __name__ == '__main__':
    main()
