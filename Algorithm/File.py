from pyspark import SparkContext
import scipy.sparse as sps
import numpy as np
import datetime
import csv
import sys
import time
import os


class File(object):
    SC = SparkContext("local", "competition")
    __PATH_PROJECT = "/Users/gengdongjie/WorkSpace/PycharmProjects/Competition/"
    __PATH_OUTPUT = None
    __PATH_TRAIN = __PATH_PROJECT + "Data/RawData/"
    __PATH_TARGET = __PATH_PROJECT + "Data/RawData/"
    __PATH_FEATURE = __PATH_PROJECT + "Data/RawData/"

    __split_raw_data = True
    __top5_dict = dict()
    __data_rdd_dict = dict()
    __feature_rdd = None
    __target_rdd = None
    __target_list = None
    __train_rdd = None
    __test_rdd = None

    def __init__(self, split_raw_data=True, self_data=False):
        self.__split_raw_data = split_raw_data
        print("[", datetime.datetime.now(), "]: Class File is initializing...")
        if self_data:
            self.__PATH_TRAIN = self.__PATH_PROJECT + "Data/SelfData/"
            self.__PATH_TARGET = self.__PATH_PROJECT + "Data/SelfData/"
            self.__PATH_FEATURE = self.__PATH_PROJECT + "Data/SelfData/"

        # prepare raw_rdd
        raw_rdd = self.SC.textFile(self.__PATH_TRAIN + 'train.csv')
        raw_header = raw_rdd.first()
        raw_rdd = raw_rdd.filter(lambda x: x != raw_header).map(lambda x: [int(i) for i in x.split(",")])
        # prepare feature_rdd
        feature_rdd = self.SC.textFile(self.__PATH_FEATURE + 'icm.csv')
        feature_header = feature_rdd.first()
        self.__feature_rdd = feature_rdd.filter(lambda x: x != feature_header).map(
            lambda x: [int(i) for i in x.split(",")])
        self.__train_rdd = raw_rdd

        if split_raw_data:
            user_item_rate = raw_rdd.map(lambda x: (x[0], {x[1]: x[2]})).reduceByKey(
                lambda x, y: dict(x.items() | y.items()))
            remain_rdd, test_rdd = user_item_rate.filter(
                lambda x: len([i for i, j in x[1].items() if j >= 8]) >= 6).randomSplit([0.8, 0.2])
            item_popularity = self.get_item_popularity()
            test_dict = test_rdd.map(
                lambda x: (x[0], {i[0]: item_popularity.index(i[0]) for i in x[1].items() if i[1] >= 8})).map(
                lambda x: (x[0], {i[0] for i in sorted(x[1].items(), key=lambda j: j[1])[:5]})).collectAsMap()
            self.__test_rdd = raw_rdd.filter(lambda x: x[0] in test_dict and x[1] in test_dict[x[0]]).cache()
            self.__train_rdd = raw_rdd.filter(lambda x: not (x[0] in test_dict and x[1] in test_dict[x[0]])).cache()
            self.__target_rdd = self.__test_rdd.map(lambda x: x[0]).distinct().sortBy(lambda x: x, ascending=True)
            self.__target_list = self.__target_rdd.collect()

            # self.output_rdd(self.__test_rdd, "test")
            # self.output_rdd(self.__train_rdd, "train")
            # self.output_rdd(self.__target_rdd, "target")

        else:
            # prepare __target_rdd
            target_rdd = self.SC.textFile(self.__PATH_TARGET + 'target_users.csv')
            target_header = target_rdd.first()
            self.__target_rdd = target_rdd.filter(lambda x: x != target_header).map(lambda x: int(x))
            self.__target_list=self.__target_rdd.collect()
        print("[", datetime.datetime.now(), "]: Class File is initialized.")

    def get_user_item_rate(self):
        user_item_rate = self.__train_rdd.map(lambda x: (x[0], {x[1]: x[2]})).reduceByKey(
            lambda x, y: dict(x.items() | y.items())).collectAsMap()
        return user_item_rate  # Dict

    def get_mean(self):
        mean = self.__train_rdd.map(lambda x: int(x[2])).mean()
        return mean  # float

    def get_users(self):
        users = self.__train_rdd.map(lambda x: int(x[0])).distinct().sortBy(lambda x: x).collect()
        return users  # List

    def get_items(self):
        items = self.__train_rdd.map(lambda x: int(x[1])).distinct().sortBy(lambda x: x).collect()
        return items  # List

    def get_target_users(self):
        return self.__target_list  # List

    def get_training_rdd(self):
        return self.__train_rdd

    def get_test_rdd(self):
        return self.__test_rdd

    def get_target_rdd(self):
        return self.__target_rdd

    def get_user_avg_rate(self):
        user_avg_rate = self.__train_rdd.map(lambda x: (x[0], {x[1]: x[2]})).reduceByKey(
            lambda x, y: dict(x.items() | y.items())).map(
            lambda x: (x[0], sum(rate for rate in x[1].values()) / len(x[1]))).collectAsMap()
        return user_avg_rate  # Dict

    def get_user_item_rate_minus_bias(self):
        user_avg_rate = self.get_user_avg_rate()
        user_item_rate_minus_bias = self.__train_rdd.map(
            lambda x: (x[0], {x[1]: x[2] - user_avg_rate[x[0]]})).reduceByKey(
            lambda x, y: dict(x.items() | y.items())).collectAsMap()
        return user_item_rate_minus_bias

    def get_item_popularity(self):
        item_popularity = self.__train_rdd.map(lambda x: (x[1], 1)).reduceByKey(
            lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False).map(lambda x: (x[0])).collect()
        return item_popularity  # List

    def get_user_seen_items(self):
        user_seen_items = self.__train_rdd.map(lambda x: (x[0], {x[1]: x[2]})).reduceByKey(
            lambda x, y: dict(x.items() | y.items())).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        return user_seen_items  # Dict

    def get_item_seen_users(self):
        item_seen_users = self.__train_rdd.map(lambda x: (x[1], {x[0]: x[2]})).reduceByKey(
            lambda x, y: dict(x.items() | y.items())).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        return item_seen_users  # Dict

    def get_item_seen_features(self):
        item_seen_features = self.__feature_rdd.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda x, y: x | y).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        return item_seen_features  # Dict

    def get_feature_seen_items(self):
        feature_seen_items = self.__feature_rdd.map(lambda x: (x[1], {x[0]})).reduceByKey(lambda x, y: x | y).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        return feature_seen_items  # Dict

    def get_user_seen_features(self):
        item_seen_features = self.get_item_seen_features()

        def get_features(item):
            item_feature = dict()
            if item in item_seen_features:
                for feature in item_seen_features[item]:
                    item_feature[feature] = 1
            return item_feature

        def acc_feature(x, y):
            intersection = x.keys() & y.keys()
            results = dict()
            for feature in x.keys() | y.keys():
                if feature in intersection:
                    results[feature] = x[feature] + y[feature]
                else:
                    results[feature] = 1
            return results

        user_seen_features = self.__train_rdd.map(
            lambda x: (x[0], get_features(x[1]))).reduceByKey(
            lambda x, y: acc_feature(x, y)).collectAsMap()
        return user_seen_features

    def get_user_pos_items(self):
        user_avg_rate = self.get_user_avg_rate()

        def rate_filter(x):
            i = set()
            if x[2] >= user_avg_rate[x[0]]:
                i.add(x[1])
            return i

        user_pos_items = self.__train_rdd.map(lambda x: (x[0], rate_filter(x)))
        user_pos_items = user_pos_items.reduceByKey(lambda x, y: x | y).collectAsMap()
        return user_pos_items

    def get_user_neg_items(self):
        user_avg_rate = self.get_user_avg_rate()

        def rate_filter(x):
            i = set()
            if x[2] < user_avg_rate[x[0]]:
                i.add(x[1])
            return i

        user_neg_items = self.__train_rdd.map(lambda x: (x[0], rate_filter(x)))
        user_neg_items = user_neg_items.reduceByKey(lambda x, y: x | y).collectAsMap()
        return user_neg_items

    def get_item_pos_users(self):
        user_avg_rate = self.get_user_avg_rate()

        def rate_filter(x):
            i = set()
            if x[2] >= user_avg_rate[x[0]]:
                i.add(x[0])
            return i

        item_pos_users = self.__train_rdd.map(lambda x: (x[1], rate_filter(x)))
        item_pos_users = item_pos_users.reduceByKey(lambda x, y: x | y).collectAsMap()
        return item_pos_users

    def get_item_neg_users(self):
        user_avg_rate = self.get_user_avg_rate()

        def rate_filter(x):
            i = set()
            if x[2] < user_avg_rate[x[0]]:
                i.add(x[0])
            return i

        item_neg_users = self.__train_rdd.map(lambda x: (x[1], rate_filter(x)))
        item_neg_users = item_neg_users.reduceByKey(lambda x, y: x | y).collectAsMap()
        return item_neg_users

    def output_rdd(self, rdd, name=""):
        if self.__PATH_OUTPUT is None:
            self.__PATH_OUTPUT = self.__PATH_PROJECT + "Output/[" + str(datetime.datetime.now().time()) + "]/"
            # noinspection PyTypeChecker
            os.mkdir(self.__PATH_OUTPUT)
        file_name = self.__PATH_OUTPUT + name + "|Rdd|.csv"
        dictionary = rdd.collectAsMap()
        bar = self.process_bar(value_max=len(dictionary), title="Output: " + name)
        with open(file_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dictionary.items():
                bar.next()
                writer.writerow([key, value])
            csv_file.close()

    def output_dict(self, dictionary, name=""):
        if self.__PATH_OUTPUT is None:
            self.__PATH_OUTPUT = self.__PATH_PROJECT + "Output/[" + str(datetime.datetime.now().time()) + "]/"
            # noinspection PyTypeChecker
            os.mkdir(self.__PATH_OUTPUT)
        file_name = self.__PATH_OUTPUT + name + "|Dict|.csv"
        bar = self.process_bar(value_max=len(dictionary), title="Output: " + name)
        with open(file_name, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, items in dictionary.items():
                bar.next()
                writer.writerow([key, items])
            csv_file.close()

    def output_train(self):
        train_dict = self.__train_rdd.map(lambda x: (x[0], {str(x[1]) + "[" + str(x[2]) + "]"})).reduceByKey(
            lambda x, y: x | y).sortBy(lambda x: x[0], ascending=True).collectAsMap()

        self.output_dict(train_dict, "train")

    def output_test(self):
        test_dict = self.__test_rdd.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda x, y: x | y).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        self.output_dict(test_dict, "test")

    def output_top5(self, func):
        if self.__PATH_OUTPUT is None:
            self.__PATH_OUTPUT = self.__PATH_PROJECT + "Output/[" + str(datetime.datetime.now().time()) + "]/"
            # noinspection PyTypeChecker
            os.mkdir(self.__PATH_OUTPUT)
        file_name = self.__PATH_OUTPUT + "Top5.csv"
        self.__top5_dict = dict()
        bar = self.process_bar(value_max=len(self.__target_list), title="Output top5")
        for user in self.__target_list:
            bar.next()
            self.__top5_dict[user] = func(user)
        with open(file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(["userId", "RecommendedItemIds"])
            for key, items in self.__top5_dict.items():
                item_str = ""
                for item in items:
                    item_str += str(item) + ' '
                writer.writerow([key, item_str])
            csv_file.close()
        if self.__split_raw_data:
            self.__print_map5()

    def __print_map5(self):
        test_dict = self.__test_rdd.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda x, y: x | y).sortBy(
            lambda x: x[0], ascending=True).collectAsMap()
        train_dict = self.__train_rdd.map(lambda x: (x[0], {str(x[1]) + "[" + str(x[2]) + "]"})).reduceByKey(
            lambda x, y: x | y).sortBy(lambda x: x[0], ascending=True).collectAsMap()
        self.output_dict(test_dict, "test")
        self.output_dict(train_dict, "train")
        print("--------Results of MAP@5---------")
        map5 = 0
        for user, items in self.__top5_dict.items():
            i = 0
            j = 0
            string = "[" + str(user) + "]"
            for item in items:
                i += 1
                if item in test_dict[user]:
                    j += 1
                    map5 += j / i
                    string += " " + str(j) + "/" + str(i) + " "
            if j > 0:
                print(string)
        map5 = map5 / 5 / len(self.__top5_dict)
        print("MAP@5=", map5)
        print("---------------------------------")

    @staticmethod
    def process_bar(value_max, title):
        class Process:
            __start_time = None
            __last_print = None
            __left_time = None
            __value_max = 0
            __value_current = 0

            __bar_length = 20
            __bar = "#" * __bar_length + " " * __bar_length

            def __init__(self, value_max_inner, title_inner):
                self.__start_time = datetime.datetime.now().replace(microsecond=0)
                self.__value_max = value_max_inner
                print(title_inner)
                line_header = '{:>6}'.format(0) + "/"'{:<6}'.format(self.__value_max)
                line_bar = "[" + self.__bar[self.__bar_length:self.__bar_length * 2] + "]"
                line_end = '{:>3}'.format(0) + "%" + " 00:00:00| ETA: 0:00:00"
                sys.stdout.write('\r' + str(line_header + line_bar + line_end))
                self.__last_print = self.__start_time

            def next(self):
                self.__value_current += 1
                now = datetime.datetime.now().replace(microsecond=0)
                percent = self.__value_current / self.__value_max
                if now - self.__last_print < datetime.timedelta(seconds=2) and percent != 1:
                    return
                percent = self.__value_current / self.__value_max
                interval = now - self.__start_time
                line_header = '{:>6}'.format(self.__value_current) + "/"'{:<6}'.format(
                    self.__value_max)
                index = int(self.__bar_length * (2 - percent))
                line_bar = "[" + self.__bar[index - self.__bar_length:  index] + "]"

                line_end = '{:>3}'.format(int(percent * 100)) + "%" + \
                           '{:>9}'.format(str(interval)) + "| ETA:" + \
                           '{:>8}'.format(str(interval * (1 - percent) / percent).split(".")[0])
                if percent == 1:
                    time.sleep(0.5)
                    line_end += " END"
                sys.stdout.write('\r' + str(line_header + line_bar + line_end))
                self.__last_print = now
                if percent == 1:
                    print("\r")
                    del self

        return Process(value_max_inner=value_max, title_inner=title)
