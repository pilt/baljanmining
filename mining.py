import cPickle as pickle
import sys
import collections

import scb.names

name_gender = {}
male_names = scb.names.male()
female_names = scb.names.female()
for names, gender in [(male_names, 'm'), (female_names, 'f')]:
    for name in names:
        name_gender[name] = gender
print "name lookup created"

with open('mining.pkl', 'rb') as f:
    d = pickle.load(f)
del f
print "dump loaded"

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

section_count = collections.defaultdict(int)
males = []
females = []
unknowns = []
user_count = len(d['users'])
for uid in d['users']:
    name = d['user_name'][uid]
    gender = name_gender.get(name, 'u')
    gender_list = dict(m=males, f=females, u=unknowns).get(gender)
    gender_list.append(uid)
    section_count[d['user_section'][uid]] += 1

print "GENDERS"
print "  males:    %s" % formatpc(males, user_count)
print "  females:  %s" % formatpc(females, user_count)
print "  unknowns: %s" % formatpc(unknowns, user_count)
print ""

print "SECTIONS"
counts = section_count.items()
counts.sort(key=lambda x: x[1])
counts.reverse()
users_with_section_count = user_count - section_count[None]
for section, count in counts:
    s = str(section).rjust(10)
    c = str(count).rjust(5)
    pc1 = formatpc(count, user_count)
    pc2 = formatpc(count, users_with_section_count)
    print "  %s: %s %s %s" % (s, c, pc1, pc2)
print ""
del counts
