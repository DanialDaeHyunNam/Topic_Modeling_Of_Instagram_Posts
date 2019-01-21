from IPython.display import display, Markdown
from collections import Counter
import re 
import numpy as np
import pandas as pd
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
import gmaps
gmaps_api = pickle.load(open("./gmaps_api.pkl", "rb"))
gmaps.configure(api_key=gmaps_api)

# text = u'This dog \U0001f602'
# print(text) # with emoji

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
# print(emoji_pattern.sub(r'', text)) # no emoji

def getAllDataFrame():
    mypath = "../insta_scrapy/insta_scrapy/scraped/"
    hashtag_dirs = [f for f in listdir(mypath + "hashtag/") if not isfile(join(mypath + "hashtag/", f))]
    location_dirs = [f for f in listdir(mypath + "location/") if not isfile(join(mypath + "location/", f))]
    display(Markdown(str(hashtag_dirs)))
    display(Markdown(str(location_dirs)))
    df_li = []
    for dir_ in hashtag_dirs:
        hash_path = mypath + "hashtag/" + dir_ + "/"
        files = []
        if dir_ == "럽스타그램":
            files = [f for f in listdir(hash_path) if isfile(join(hash_path, f)) and ("csv" in f) and ("new" in f)]
        else:
            files = [f for f in listdir(hash_path) if isfile(join(hash_path, f)) and "csv" in f]
        for file in files:   
            file_path = hash_path + file
            df = pd.read_csv(file_path)
            display(Markdown(file + " : " + str(len(df))))
            df_li.append(df)
            
    for dir_ in location_dirs:
        location_dirs = mypath + "location/" + dir_ + "/"
        files = [f for f in listdir(location_dirs) if isfile(join(location_dirs, f)) and "csv" in f]
        for file in files:    
            file_path = location_dirs + file
            df = pd.read_csv(file_path)
            display(Markdown(file + " : " + str(len(df))))
            df_li.append(df)
    print(len(df_li))
    return df_li

def saveModelObjectAsPickle(model, fileName):
    joblib.dump(model, fileName)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx)) 
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])) 
        
def __list_to_str(li):
    result = ""
    for tmp in li:
        result += tmp + " "
    return result

def __getTags(data):
    li = []
    if type(data) == str:
        li = [ re.sub("\W^[ ]", "", tmp.strip()) for tmp in data.replace("#\n", "").split("#")[1:] ]
        li = [ tmp.strip().split(" ")[0] for tmp in li]
        li = [ tmp.split("\n")[0] for tmp in li]        
        li = [ re.sub("\W", "", tmp.strip()) for tmp in li]
#         return ", ".join(li) + ", "
        return li
    else:
#         return ", "
        return []

def __leaveCapionOlny(data):
    li = []
    if type(data) == str:
        li = [ re.sub("\W^[ ]", "", tmp.strip()) for tmp in data.replace("#\n", "").split("#")[1:] ]
        li = [ tmp.strip().split(" ")[0] for tmp in li]
        li = [ tmp.split("\n")[0] for tmp in li]        
        li = [ re.sub("\W", "", tmp.strip()) for tmp in li]
        for tmp in li:
            tmp = "#" + tmp
            data = data.replace(tmp, " ")
            data = data.replace("\n", " ")
        return emoji_pattern.sub(r'', data).replace("\n", " ")
    else:
        return data

def __getTagCount(data):
    if data.split(", ")[0] == "":
        return len(data.split(", ")[1:-1])
    return len(data.split(", ")[:-1])

def __list_to_str(li):
    result = ""
    for tmp in li:
        result += tmp + " "
    return result
    
def makeOneTrainDf(df_li):
    print("Df list length : {}".format(len(df_li)))
    df_cu = df_li[0]
    for idx in range(len(df_li) - 1):
        next_idx = idx + 1
        if next_idx <= len(df_li) - 1:
            df_nx = df_li[next_idx]
            mask = np.intersect1d(df_cu.id.values, df_nx.id.values)
            df_to_concat = df_nx[~df_nx["id"].isin(np.intersect1d(df_cu.id.values, df_nx.id.values))]
            print("Df length changes after concat (if 0 means all datas are unique) : " + str(len(df_nx) - len(df_to_concat)))
            df_cu = pd.concat([df_cu, df_to_concat])
    print("df_train shape : {}".format(df_cu.shape))
    return df_cu.reset_index(drop=True)

def __getDfWithLoaction(df):
    # taken_at_timestamp is a unix timestamp - we add a column for the date-time
    time = '2018-01-01'
    df['taken_at_timestamp'] = df['taken_at_timestamp'].astype(int)
    df['time'] = pd.to_datetime(df['taken_at_timestamp'], unit='s')
    filtered = df[df['loc_id']!=0]
    filtered = filtered.sort_values(by=['time'], ascending=False)
    filtered = filtered[(filtered['time'] > time)]
    return filtered

def __makePopUpList(df):
    result = []
    for idx in range(len(df)):
        df_ = df.iloc[idx]
        result.append({
          "owner_name" : df_.owner_name,
          "is_video" : df_.is_video,
          "likes" : df_.likes,
          "time" : df_.time,
          "loc_name" : df_.loc_name,
          "location" : (df_.loc_lat, df_.loc_lon),
          "comment_cnt" : df_.comment_cnt,
          "video_view_count" : df_.video_view_count,
          "caption" : df_.caption,
          "tags" : ("#" + "#".join(df_.tags)) if df_.tags_cnt != 0 else None,
        })
    return result

def drawMarkerOnMap(df):
    df_ = __getDfWithLoaction(df)
    popUpList = __makePopUpList(df_)
    post_locations = [post['location'] for post in popUpList]

    info_box_template = """
    <dl>
    <dt>User</dt><dd>{owner_name}</dd>
    <dt>Is_Video</dt><dd>{is_video}</dd>
    <dt>Likes</dt><dd>{likes}</dd>
    <dt>Time</dt><dd>{time}</dd>
    <dt>Location Name</dt><dd>{loc_name}</dd>
    <dt>Comment Count</dt><dd>{comment_cnt}</dd>
    <dt>View</dt><dd>{video_view_count}</dd>
    <dt>Tags</dt><dd>{tags}</dd>
    </dl>
    """
    post_info = [info_box_template.format(**post) for post in popUpList]
    marker_layer = gmaps.marker_layer(post_locations, info_box_content=post_info)
    fig = gmaps.figure(center=(37.532600, 127.024612), zoom_level=10)
    fig.add_layer(marker_layer)
    
    print(len(post_info))
    return fig

def make_df_i_want(df):
    df_train = df.copy()
    df_train["tags"] = df_train.caption.apply(__getTags)
    df_train["tags_from_comment"] = df_train.first_comment.apply(__getTags)
    df_train["tag_total"] = df_train["tags"] + df_train["tags_from_comment"]
    df_train["tags_cnt"] = df_train.tag_total.apply(lambda a: len(a))
    df_train.drop(["tags", "tags_from_comment"], axis = 1, inplace=True)
    df_train.rename({"tag_total" : "tags"}, axis=1, inplace=True)
    df_train["caption_only"] = df_train.caption.apply(__leaveCapionOlny)
    df_train["tags_str"] = df_train.tags.apply(__list_to_str)
    df_train["duplicated_tag"] = df_train.tags.apply(lambda a: 1 if len(a) != len(set(a)) else 0) 
    return df_train

def get_counter(df_train, search_word = ""):
    count_search = Counter()
    userTags = df_train["tags"].values
    for tag in userTags:
        if search_word in tag:
            count_search.update(tag)
    return count_search

def saveDf(df, filename):
    filename = "./asset/" + filename + ".csv"
    df.to_csv(filename, index=False)