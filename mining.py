import cPickle as pickle
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

WORKER = 'worker'
ONCALL = 'oncall'
SIGNUP = 'signup'
BUYER = 'buyer'
ALL = 'all'
MALE = 'male'
FEMALE = 'female'
UNKNOWN = 'unknown'
GENDERS = [MALE, FEMALE, UNKNOWN]
CLASSES = [WORKER, ONCALL, SIGNUP, BUYER, ALL] + GENDERS


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
        key CHARACTER VARYING(20) PRIMARY KEY,
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
        """INSERT INTO classes (key, name) VALUES ("%s", "%s");""" % (g, g))

def do_queries(queries):
    for query in queries:
        try:
            c.execute(query)
        except sqlite3.OperationalError:
            print query
            raise


if settings.DO_IMPORT:
    do_queries(import_schema)
    conn.commit()

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


class Hash(object):

    def __init__(self, row):
        for idx, col in enumerate(row.keys()):
            setattr(self, col, row[idx])

    def __str__(self):
        return str(self.__dict__)


def hashes(res):
    return [Hash(r) for r in res]


@memoized
def get_interval(start, end):
    c.execute(
        "SELECT * FROM intervals WHERE start = ? AND end = ? LIMIT 1",
        (start, end))
    return Hash(c.fetchone())


@memoized
def buys_within(start, end):
    interval = get_interval(start, end)
    c.execute(
        "SELECT * FROM interval_buys WHERE interval = ?;",
        (interval.id,))
    return hashes(c.fetchall())


@memoized
def user_buys_within(user, start, end):
    interval = get_interval(start, end)
    c.execute(
        """SELECT count(*) as count FROM interval_buys
            WHERE interval = ? AND user = ?;""",
            (interval.id, user.id))
    return Hash(c.fetchone()).count


@memoized
def shifts_within(start, end):
    interval = get_interval(start, end)
    c.execute(
        "SELECT * FROM interval_shifts WHERE interval = ?;""",
        (interval.id,))
    return hashes(c.fetchall())


@memoized
def signups_within(start, end, g=None):
    interval = get_interval(start, end)
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
def get_users(iterable):
    user_ids = set()
    for i in iterable:
        user_ids.add(str(i.user))
    c.execute(
        "SELECT * FROM users WHERE id IN (%s);" % ", ".join(user_ids))
    return hashes(c.fetchall())


@memoized
def with_gender(gender, users):
    assert gender in GENDERS
    return filter(lambda user: user.gender == gender, users)


@memoized
def class_within(g, start, end):
    if g in [WORKER, ONCALL]:
        return get_users(signups_within(start, end, g))
    elif g == SIGNUP:
        return get_users(signups_within(start, end))
    elif g == BUYER:
        return get_users(buys_within(start, end))
    elif g == ALL:
        iterable = []
        iterable.extend(signups_within(start, end))
        iterable.extend(buys_within(start, end))
        return get_users(iterable)
    elif g in GENDERS:
        return with_gender(g, class_within(ALL, start, end))
    assert False, g


@memoized
def classes_within(start, end):
    classes = {}
    for g in CLASSES:
        classes[g] = class_within(g, start, end)
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
    return key not in GENDERS
def get_gendered(obj):
    return traverse(obj, _gender_node, _gender_test)


@memoized
def _interval_buy_node(interval, users):
    return [user_buys_within(user, interval.start, interval.end) for user in users]
def get_buys(obj, interval):
    func = functools.partial(_interval_buy_node, interval)
    return traverse(obj, func)


@memoized
def format_counts(obj, indent=0):
    s = ""
    i = " " * indent
    if isinstance(obj, dict):
        for key, val in obj.items():
            s += i + str(key) + '\n' + format_counts(val, indent+2)
    else:
        count = len(obj)
        s += i + str(count) + '\n'
    return s


def get_counted(obj):
    return traverse(obj, len)


def get_summed(obj):
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

c.execute("SELECT * FROM semesters ORDER BY start;")
semesters = hashes(c.fetchall())

if settings.DO_EXTRAS:
    do_queries(extra_schema)
    conn.commit()

    for sem in semesters:
        insert_interval(sem.name, sem.start, sem.end)
    conn.commit()


from pprint import pprint


def transform(base, steps):
    new = copy.deepcopy(base)
    for step in steps:
        new = step(new)
    return new


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

metrics = [BUYER, WORKER, ONCALL]
outs = ['blips.csv', 'people.csv']
for csv_file in outs:
    fields = ['semester']
    for metric in metrics:
        for gender in [ALL] + GENDERS:
            fields.append("%s_%s" % (metric, gender))
    with open(csv_file, 'w') as csv:
        csv.write('')
    with open(csv_file, 'a+') as csv:
        line = ", ".join(map(str, fields))
        print line
        csv.write(line + '\n')

for sem in semesters:
    genders = get_gendered
    buys = lambda obj: get_buys(obj, sem)
    sums = get_summed
    counts = get_counted
    classes = classes_within(sem.start, sem.end)

    datas = [
        transform(classes, (genders, buys, sums)),
        transform(classes, (genders, counts)),
        ]
    for csv_file, data in zip(outs, datas):
        if data[ALL][ALL] == 0:
            continue
        fields = []
        fields.append(sem.name)
        for metric in metrics:
            for gender in [ALL] + GENDERS:
                fields.append(data[metric][gender])
        with open(csv_file, 'a+') as csv:
            line = ",".join(map(str, fields))
            print line
            csv.write(line + '\n')


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
