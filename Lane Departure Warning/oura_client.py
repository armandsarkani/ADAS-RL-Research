import json
import urllib.request
import datetime
import certifi
import ssl
import os

# URL access variables
access_token = 'WBR5SYQD7US3GG6BYNAAKKGC3MGPAS6G'
url_userinfo = 'https://api.ouraring.com/v1/userinfo?access_token='
url_sleep = 'https://api.ouraring.com/v1/sleep?access_token='
url_activity = 'https://api.ouraring.com/v1/activity?access_token='
url_readiness = 'https://api.ouraring.com/v1/readiness?access_token='
start_tag = '&start='
end_tag = '&end='

# date/time variables
today = datetime.date.today()
yesterday = today - datetime.timedelta(days = 1)
today = str(today)
yesterday = str(yesterday)

# functions to receive Oura ring data
def get_userinfo(start=yesterday, end=today):
    url = url_userinfo + access_token + start_tag + start + end_tag + end
    data = urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())).read().decode()
    obj = json.loads(data)
    return obj
def get_sleep(start=yesterday, end=today):
    url = url_sleep + access_token + start_tag + start + end_tag + end
    data = urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())).read().decode()
    obj = json.loads(data)
    return obj
def get_activity(start=yesterday, end=today):
    url = url_activity + access_token + start_tag + start + end_tag + end
    data = urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())).read().decode()
    obj = json.loads(data)
    return obj
def get_readiness(start=yesterday, end=today):
    url = url_readiness + access_token + start_tag + start + end_tag + end
    data = urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())).read().decode()
    obj = json.loads(data)
    return obj
def get_all_data():
    userinfo = get_userinfo()
    sleep = get_sleep()
    activity = get_activity()
    readiness = get_readiness()
    return userinfo, sleep, activity, readiness
    
# helper functions
def print_json():
    userinfo = get_userinfo()
    sleep = get_sleep()
    activity = get_activity()
    readiness = get_readiness()
    print("User info:")
    print(userinfo)
    print("\nSleep:")
    print(sleep)
    print("\nActivity:")
    print(activity)
    print("\nReadiness:")
    print(readiness)
def save_json():
    if(not(os.path.exists('Oura Data'))):
        os.mkdir('Oura Data')
    os.chdir('Oura Data')
    userinfo_outfile = open('UserInfo.json', 'w')
    sleep_outfile = open('Sleep.json', 'w')
    activity_outfile = open('Activity.json', 'w')
    readiness_outfile = open('Readiness.json', 'w')
    userinfo, sleep, activity, readiness = get_all_data()
    json.dump(userinfo, userinfo_outfile)
    json.dump(sleep, sleep_outfile)
    json.dump(activity, activity_outfile)
    json.dump(readiness, readiness_outfile)
