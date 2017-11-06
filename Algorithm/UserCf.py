from Algorithm import File
import math
import operator
import random

f = File.File(split_raw_data=False)

USERS = f.get_users()
ITEMS = f.get_items()
USER_SEEN_ITEMS = f.get_user_seen_items()
ITEM_SEEN_USERS = f.get_item_seen_users()
ITEM_POPULARITY = f.get_item_popularity()
USER_ITEM_RATE = f.get_user_item_rate()
# build userSimilarity
# calculate co-rated items between users

count = dict()

# calculate co-rated items between users
bar = f.process_bar(value_max=len(ITEM_SEEN_USERS), title="STEP: [1/3]")
for item, users in ITEM_SEEN_USERS.items():
    bar.next()
    for user1 in users:
        for user2 in users:
            if user1 == user2:
                continue
            if user1 in count:
                if user2 in count[user1]:
                    count[user1][user2] += 1 / math.log(1 + len(users))
                else:
                    count[user1].update({user2: 1 / math.log(1 + len(users))})
            else:
                count[user1] = {}
                count[user1].update({user2: 1 / math.log(1 + len(users))})

# calculate final similarity
userSimilarity = dict()
bar = f.process_bar(value_max=len(count), title="STEP: [2/3]")
for user1, related_users in count.items():
    bar.next()
    userSimilarity[user1] = {}
    for user2, num in related_users.items():
        a = math.sqrt(len(USER_SEEN_ITEMS[user1]) * len(USER_SEEN_ITEMS[user2]))
        userSimilarity[user1].update({user2: num / a})

# function3
recommendationRank = dict()


def set_knn(knn=10):
    bar = f.process_bar(value_max=len(userSimilarity), title="STEP: [3/3]")
    recommendationRank.clear()
    for user1 in userSimilarity.keys():
        bar.next()
        recommendationRank[user1] = {}
        seen_items = USER_SEEN_ITEMS[user1]
        for user2, simValue in sorted(userSimilarity[user1].items(), key=operator.itemgetter(1), reverse=True)[:knn]:
            for user2Item in USER_SEEN_ITEMS[user2]:
                if user2Item in seen_items:
                    # filter items user interacted before
                    continue
                if user2Item in recommendationRank[user1]:
                    recommendationRank[user1][user2Item] += simValue * 1

                else:
                    recommendationRank[user1].update({user2Item: simValue * 1})


# function4
def recommend_top5(user):
    top5 = []
    if user in recommendationRank:
        for item, score in sorted(recommendationRank[user].items(), key=operator.itemgetter(1), reverse=True)[0:5]:
            top5.append(item)
    while len(top5) < 5:
        top5.append(ITEM_POPULARITY[random.choice(range(100))])
    return top5


set_knn(240)
f.output_top5(recommend_top5)


