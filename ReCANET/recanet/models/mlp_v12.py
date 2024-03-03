from .nbr_base import NBRBase
from utils.metrics import *
import os
seed_value = 12321
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.layers import Input,multiply ,Dense, Dropout, Embedding,Concatenate, Reshape,Flatten,LSTM, Attention, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


class MLPv12(NBRBase):
    def __init__(self, train_baskets, test_baskets,valid_baskets,dataset ,basket_count_min=0, min_item_count = 5, user_embed_size = 32,item_embed_size = 128,h1 = 128,h2 = 128,h3 = 128,h4 = 128,h5 = 128,history_len = 40, job_id = 1):
        super().__init__(train_baskets,test_baskets,valid_baskets,basket_count_min)
        self.model_name = dataset+ 'simple_mlpv12'
        self.dataset = dataset
        self.all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        self.all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        self.num_items = len(self.all_items) +1
        self.num_users = len(self.all_users) +1

        print("items:", self.num_items)
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        item_counts = item_counts[item_counts['item_count']>= min_item_count]
        item_counts_dict = dict(zip(item_counts['item_id'],item_counts['item_count']))
        print("filtered items:", len(item_counts_dict))
        self.num_items = len(item_counts_dict) +1
        self.item_id_mapper = {}
        self.id_item_mapper = {}
        self.user_id_mapper = {}
        self.id_user_mapper = {}
      
        counter = 0
        for i in range(len(self.all_items)):
            if self.all_items[i] in item_counts_dict:
                self.item_id_mapper[self.all_items[i]] = counter+1
                self.id_item_mapper[counter+1] = self.all_items[i]
                counter+=1
        for i in range(len(self.all_users)):
            self.user_id_mapper[self.all_users[i]] = i+1
            self.id_user_mapper[i+1] = self.all_users[i]

        self.user_embed_size = user_embed_size#32
        self.item_embed_size = item_embed_size#128
        #self.hidden_size = hidden_size#128
        self.history_len = history_len#40
        self.num_layers = 3

        self.data_path = self.model_name+'_'+str(job_id) + '_' + str(self.user_embed_size) + '_' + \
                         str(self.item_embed_size) + '_'+ str(h1) + '_'+ str(h2) + '_'+ str(h3) + '_'+ str(h4) + '_'+ str(h5) + '_' + str(self.history_len) + '_' + \
                         str(self.num_layers)

        input1 = Input(shape=(1,))
        input2 = Input(shape=(1,))
        input3 = Input(shape=(self.history_len,))
        input4 = Input(shape=(self.history_len,))
        i5 = Input(shape=(self.history_len,))

        x1 = Embedding(self.num_items, self.item_embed_size , input_length=1)(input1)
        x2 = Embedding(self.num_users, self.user_embed_size, input_length=1)(input2)
        
        x1 = Flatten()(x1)
        x2 = Flatten()(x2)
        x3 = input3
        x4 = input4
        x5 = i5

        x11 = Dense(h1, activation= 'relu')(Concatenate()([x1,x2]))
        x12 = tf.keras.layers.RepeatVector(self.history_len)(x11)
        x14 = Reshape((self.history_len,1))(x4)
        x15 = Reshape((self.history_len,1))(x5)
        x14 = Dense(h1, activation= 'relu')(Concatenate()([x12,x14,x15]))

        x = LSTM(h2,return_sequences = True)(x14, mask = tf.dtypes.cast(input4, tf.bool))
        x = LSTM(h3)(x, mask = tf.dtypes.cast(input4, tf.bool))


        x = Dense(h4, activation='relu')(x)
        x = Dense(h5, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        self.model = Model([input1,input2,input3,input4,i5], output)

    def create_train_data(self):
        print(self.data_path)
        if os.path.isfile(self.model_name +'_' + str(self.history_len) + '_train_users.npy'):
            train_users = np.load(self.model_name +'_' + str(self.history_len) + '_train_users.npy')
            train_items = np.load(self.model_name +'_' + str(self.history_len) + '_train_items.npy')
            train_history = np.load(self.model_name +'_' + str(self.history_len) + '_train_history.npy')
            train_history2 = np.load(self.model_name +'_' + str(self.history_len) + '_train_history2.npy')
            train_labels = np.load(self.model_name +'_' + str(self.history_len)+ '_train_labels.npy')
            return train_items,train_users, train_history,train_history2, train_labels

        basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()

        
        
        basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['item_id']))
        basket_items_dict['null'] = []

        user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates().\
            sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()

        user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['basket_id']))


        train_users = []
        train_items = []
        train_history = []
        train_history2 = []
        rr = []
        train_labels = []
        
        print('num users:', len(self.test_users))
        print('====')
        print(self.num_users)
        print('====')

        cnt={}

        data11 = []

        #####################################
        import pandas as pd
        import numpy as np
        #import matplotlib.pyplot as plt
        #from scipy.stats import norm

        df = pd.read_csv('sampled_data.csv')
        
        # Get unique item IDs from the DataFrame
        unique_item_ids = df['item_id'].unique()
        
        # Dictionary to store item order count vectors
        item_order_count_dict = {}

        pr = {}
        uniq = {}
        org = {}
        final={}
        final_percentile={}
        
        # Iterate through unique item IDs
        for item_id in unique_item_ids:
            # Filter the DataFrame for the specified item ID
            item_counts = df[df['item_id'] == item_id].groupby('user_id').size().reset_index(name='order_count')
        
            if not item_counts.empty:
                
                # Extract the order counts as a NumPy array
                iv = item_counts['order_count'].values
                item_order_count_dict[item_id] = iv

                # Calculate the unique elements and their occurrences
                total_elements = len(iv)
                unique_elements, element_counts = np.unique(iv, return_counts=True)
                org[item_id] = iv
                uniq[item_id] = unique_elements
                
                
                # Generate the probability distribution
                probability_distribution = element_counts / total_elements
                
                # Round off the probabilities to two decimal places
                rounded_probabilities = np.round(probability_distribution, 2)
                
        
                x = unique_elements
                prob = element_counts
                pr[item_id] = rounded_probabilities
                
                # Create an array of values based on the provided data and probabilities
                data = np.repeat(x, np.round(np.array(prob) * 100).astype(int))
                
                # Calculate the mean and standard deviation using the provided data
                mu = np.average(data)
                std_dev = np.std(data)
                
                # Generate samples from a normal distribution with the calculated mean and standard deviation
                samples = np.random.normal(mu, std_dev, size=1000)
                
                # Specify the desired percentiles
                percentiles = np.linspace(16, 86, 20)
                
                # Calculate values at specified percentiles
                percentile_values = np.percentile(samples, percentiles)
                
                # Round off the values and take the absolute value
                #rounded_list = [round(abs(element)) for element in percentile_values]

                rounded_list=[]
                for it in percentile_values:
                    if it<=0:
                        rounded_list.append(0)
                    else:
                        rounded_list.append(round(it))

                final[item_id]=rounded_list
                final_percentile[item_id]=percentiles




                #plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Generated Normal Distribution')
                #plt.plot(unique_elements, norm.pdf(unique_elements, mu, std_dev), 'r-', lw=2, label='Fitted Normal Distribution')
                #plt.scatter(percentile_values, norm.pdf(percentile_values, mu, std_dev), color='blue', label='Percentile Values')
                #plt.title('Generated and Fitted Normal Distribution with Percentile Values')
                #plt.xlabel('Values')
                #plt.ylabel('Probability Density')
                #plt.legend()
                #plt.show()



        
        #####################################





        
        for c,user in enumerate(self.test_users):
            if c % 1000 ==1:
                print(c , 'user passed')

            baskets = user_baskets_dict[user]
            item_seq = {}

            #print('=====')
            #7  ->  [27255010658, 27266821687, 27353377196, 27505412356] => [0, 1, 2, 3]
            #print(user,' -> ',baskets)
            #print('=====')
            
            # this loop count the number of basket  i contain count number and basket contain basket id
            for i, basket in enumerate(baskets):

                for item in basket_items_dict[basket]:

                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)

            #print('====')
            # user -> 7 item_seq  ->  {865569: [0], 886703: [0], 889731: [0], 893400: [0], 995436: [0], 1020581: [0], 1022003: [0, 2], 1029743: [0], 1048440: [0], 1074172: [0], 1077373: [0], 1082185: [0, 2], 1085846: [0], 1122358: [0, 2], 1128333: [0, 2], 6391497: [0], 9296919: [0], 9487927: [0], 1011876: [1], 856335: [2], 862682: [2], 864615: [2], 896369: [2], 908940: [2], 921554: [2], 930917: [2], 938700: [2], 961554: [2], 979779: [2], 987724: [2], 998270: [2], 1041297: [2], 1042438: [2], 1044078: [2], 1067849: [2], 1085604: [2], 1092045: [2], 1124352: [2], 1132198:[2], 5568378: [2], 5590695: [2], 6944571: [2]}
            #print('user ->',user,'item_seq',' -> ',item_seq)
            #print('====')

            
            for i in range(max(0,len(baskets)-50), len(baskets)):
                label_basket = baskets[i]
                all_history_baskets = baskets[:i]
                items = []

                #print('=====')
                #print('user -> ',user,' i -> ',i,' ,basket[i]-> ',baskets[i])
                #print('basket[:i] -> ',all_history_baskets)

                '''
                =====
                user ->  7  i ->  0  ,basket[i]->  27255010658
                basket[:i] ->  []
                =====
                =====
                user ->  7  i ->  1  ,basket[i]->  27266821687
                basket[:i] ->  [27255010658]
                =====
                =====
                user ->  7  i ->  2  ,basket[i]->  27353377196
                basket[:i] ->  [27255010658, 27266821687]
                =====
                =====
                user ->  7  i ->  3  ,basket[i]->  27505412356
                basket[:i] ->  [27255010658, 27266821687, 27353377196]
                =====
                '''
                #print('=====')

                #items contain all the item which is ordered in the past history baskets
                for basket in all_history_baskets:
                    for item in basket_items_dict[basket]:
                        items.append(item)
                items = list(set(items))
                #print('========')
                #print('user -> ',user,' i -> ',i,' ,basket[i]-> ',baskets[i])
                #print('items ->',items)
                #print('item in ',baskets[i],' -> ',basket_items_dict[baskets[i]])
                #print('========')
                
                for item in items:
                    if item not in self.item_id_mapper:
                        continue
                    index = np.argmax(np.array(item_seq[item])>=i)
                    if np.max(np.array(item_seq[item])) < i:
                        index = len(item_seq[item])
                    input_history = item_seq[item][:index].copy()
                    
                    if len(input_history) ==0:
                        continue
                    if len(input_history) ==1 and input_history[0]==-1:
                        continue

                    
                    while len(input_history) < self.history_len:
                        input_history.insert(0,-1)
                        
                    real_input_history = []
                    
                    for x in input_history:
                        if x == -1:
                            real_input_history.append(0)
                        else:
                            real_input_history.append(i-x)
                            
                    real_input_history2 = []
                    for j,x in enumerate(input_history[:-1]):
                        if x == -1:
                            real_input_history2.append(0)
                        else:
                            real_input_history2.append(input_history[j+1]-input_history[j])
                            
                    real_input_history2.append(i-input_history[-1])

                    #real_input_history2 = np.concatenate((final[item], real_input_history2[-10:]))
                    
                    train_users.append(self.user_id_mapper[user])
                    train_items.append(self.item_id_mapper[item])
                    train_history.append(real_input_history[-self.history_len:])
                    train_history2.append(real_input_history2[-self.history_len:])
                    rr.append(final[item])

                    
                    user_data=np.array(user)
                    item_data=np.array(item)
                    history1=np.array(real_input_history2[-self.history_len:])
                    derived_history=np.array(final[item])
                    item_percentile=np.array(final_percentile[item])
                    

                    data11.append([user_data,item_data,history1,derived_history,item_percentile])
                    

                    #print(item, basket_items_dict[label_basket])
                    #train_labels.append(float(item in basket_items_dict[label_basket]))
                    #print('----------------')
                    #print(user," item-> ",item," i-> ",i," basket ->",baskets[i])
                    #print('input history ->',input_history)
                    #print('real_input_history[] -> ',real_input_history[-self.history_len:])
                    #print('real_input_history2[] -> ',real_input_history2[-self.history_len:])
                    #print('lable -> ',float(item in basket_items_dict[label_basket]))
                    #print('original item vector -> ',org[item])
                    #print('item vector -> ', uniq[item])
                    #print('probability vector -> ',pr[item])
                    #print('new global vector -> ', final[item])
                    #print('----------------')
                    if user not in cnt:
                        cnt[user]=1
                    else:
                        cnt[user]=cnt[user]+1

        #print('user 7 =>>>>> ', cnt[7])
        train_items = np.array(train_items)
        train_users = np.array(train_users)
        train_history = np.array(train_history)
        rr = np.array(rr)#########
        train_history2 = np.array(train_history2)
        train_labels = np.array(train_labels)
        random_indices = np.random.choice(range(len(train_items)), len(train_items),replace=False)
        train_items = train_items[random_indices]
        train_users = train_users[random_indices]
        
        train_history = train_history[random_indices]
        train_history2 = train_history2[random_indices]

        print('------')
        #print('rr ->',len(rr))
        #print('train history  ->',len(train_history))
        print('------')
        
        #train_labels = train_labels[random_indices]  

        #----create csv file-------
        df = pd.DataFrame(data11, columns=['user_data', 'item_data', 'history1', 'derived_history','item_percentile'])
        
        # Save DataFrame to CSV
        df.to_csv('final.csv', index=False)
        ##--------------------

        np.save(self.model_name +'_' + str(self.history_len) + '_train_items.npy',train_items)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_users.npy',train_users)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_history.npy',train_history)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_history2.npy',train_history2)
        np.save(self.model_name +'_' + str(self.history_len) + '_train_labels.npy',train_labels)

        return train_items,train_users, train_history,train_history2,rr ,train_labels

    def train(self):
        train_items, train_users, train_history,train_history2,rr, train_labels = self.create_train_data()
        print(train_history.shape)
        print(np.count_nonzero(train_labels))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.data_path+'_weights.{epoch:02d}.hdf5',
            save_weights_only=True,
            save_best_only=False)

        self.model.compile(loss='binary_crossentropy',#'mean_squared_error',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        print(self.model.summary())
        history = self.model.fit([train_items,train_users,train_history,train_history2,rr],train_labels, validation_split = None,
                                 batch_size=10000, epochs=5,shuffle=True, callbacks=[model_checkpoint_callback])#, class_weight= {0:1, 1:100})
        print("Training completed")

    def create_test_data(self,test_data='test'):
        if os.path.isfile(self.model_name +'_' + str(self.history_len)+ '_'+test_data+'_users.npy'):
            test_users = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_users.npy')
            test_items = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_items.npy')
            test_history = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history.npy')
            test_history2 = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history2.npy')
            test_labels = np.load(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_labels.npy')
            return test_items,test_users, test_history,test_history2, test_labels
        train_basket_items = self.train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        train_basket_items_dict = dict(zip(train_basket_items['basket_id'],train_basket_items['item_id']))

        train_user_baskets = self.train_baskets[['user_id','date','basket_id']].drop_duplicates(). \
            sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
        train_user_baskets_dict = dict(zip(train_user_baskets['user_id'],train_user_baskets['basket_id']))

        train_user_items = self.train_baskets[['user_id','item_id']].drop_duplicates().groupby(['user_id'])['item_id'] \
            .apply(list).reset_index()
        train_user_items_dict = dict(zip(train_user_items['user_id'],train_user_items['item_id']))

        test_user_items = None
        if test_data == 'test':
            test_user_items = self.test_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        else:
            test_user_items = self.valid_baskets.groupby(['user_id'])['item_id'].apply(list).reset_index()
        test_user_items_dict = dict(zip(test_user_items['user_id'],test_user_items['item_id']))

        test_users = []
        test_items = []
        test_history = []
        test_history2 = [] 
        test_labels = []
        rr = []

        #####################################
        import pandas as pd
        import numpy as np
        #import matplotlib.pyplot as plt
        #from scipy.stats import norm

        df = pd.read_csv('test_baskets.csv')
        
        # Get unique item IDs from the DataFrame
        unique_item_ids = df['item_id'].unique()
        
        # Dictionary to store item order count vectors
        item_order_count_dict = {}

        pr = {}
        uniq = {}
        org = {}
        final={}
        
        # Iterate through unique item IDs
        for item_id in unique_item_ids:
            # Filter the DataFrame for the specified item ID
            item_counts = df[df['item_id'] == item_id].groupby('user_id').size().reset_index(name='order_count')
        
            if not item_counts.empty:
                
                # Extract the order counts as a NumPy array
                iv = item_counts['order_count'].values
                item_order_count_dict[item_id] = iv

                


                # Calculate the unique elements and their occurrences
                total_elements = len(iv)
                unique_elements, element_counts = np.unique(iv, return_counts=True)
                org[item_id] = iv
                uniq[item_id] = unique_elements
                
                
                # Generate the probability distribution
                probability_distribution = element_counts / total_elements
                
                # Round off the probabilities to two decimal places
                rounded_probabilities = np.round(probability_distribution, 2)
                
        
                x = unique_elements
                prob = element_counts
                pr[item_id] = rounded_probabilities
                
                # Create an array of values based on the provided data and probabilities
                data = np.repeat(x, np.round(np.array(prob) * 100).astype(int))
                
                # Calculate the mean and standard deviation using the provided data
                mu = np.average(data)
                std_dev = np.std(data)
                
                # Generate samples from a normal distribution with the calculated mean and standard deviation
                samples = np.random.normal(mu, std_dev, size=1000)
                
                # Specify the desired percentiles
                percentiles = np.linspace(0, 100, 20)
                
                # Calculate values at specified percentiles
                percentile_values = np.percentile(samples, percentiles)
                
                # Round off the values and take the absolute value
                #rounded_list = [round(abs(element)) for element in percentile_values]

                rounded_list=[]
                for it in percentile_values:
                    if it<=0:
                        rounded_list.append(0)
                    else:
                        rounded_list.append(round(it))

                final[item_id]=rounded_list




                #plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Generated Normal Distribution')
                #plt.plot(unique_elements, norm.pdf(unique_elements, mu, std_dev), 'r-', lw=2, label='Fitted Normal Distribution')
                #plt.scatter(percentile_values, norm.pdf(percentile_values, mu, std_dev), color='blue', label='Percentile Values')
                #plt.title('Generated and Fitted Normal Distribution with Percentile Values')
                #plt.xlabel('Values')
                #plt.ylabel('Probability Density')
                #plt.legend()
                #plt.show()



        
        #####################################





        

        train_basket_items_dict['null'] = []
        for c,user in enumerate(test_user_items_dict):
            if user not in train_user_baskets_dict:
                continue
            if c % 100 ==1:
                print(c , 'user passed')
                #break

            baskets = train_user_baskets_dict[user]
            item_seq = {}
            for i, basket in enumerate(baskets):
                for item in train_basket_items_dict[basket]:
                    if item not in self.item_id_mapper:
                        continue
                    if item not in item_seq:
                        item_seq[item] = []
                    item_seq[item].append(i)


            label_items = test_user_items_dict[user]

            items = list(set(train_user_items_dict[user]))

            #print(len(history_baskets))
            for item in items:#train_user_items_dict[user]:
                if item not in self.item_id_mapper:
                    continue
                input_history = item_seq[item][-self.history_len:]
                if len(input_history) ==0:
                    continue
                if len(input_history) ==1 and input_history[0]==-1:
                    continue
                while len(input_history) < self.history_len:
                    input_history.insert(0,-1)
                real_input_history = []
                for x in input_history:
                    if x == -1:
                        real_input_history.append(0)
                    else:
                        real_input_history.append(len(baskets)-x)

                real_input_history2 = []
                for j,x in enumerate(input_history[:-1]):
                    if x == -1:
                        real_input_history2.append(0)
                    else:
                        real_input_history2.append(input_history[j+1]-input_history[j])
                        
                real_input_history2.append(len(baskets)-input_history[-1])
                test_users.append(self.user_id_mapper[user])
                test_items.append(self.item_id_mapper[item])
                test_history.append(real_input_history)
                test_history2.append(real_input_history2)
                test_labels.append(float(item in label_items))

                if item in final:
                    rr.append(final[item])
                else:
                    xx=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    rr.append(xx)

        test_items = np.array(test_items)
        test_users = np.array(test_users)
        test_history = np.array(test_history)
        test_history2 = np.array(test_history2)
        test_labels = np.array(test_labels)
        rr = np.array(rr)

        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_items.npy',test_items)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_users.npy',test_users)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history.npy',test_history)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_history2.npy',test_history2)
        np.save(self.model_name +'_' + str(self.history_len) + '_'+test_data+'_labels.npy',test_labels)

        return test_items,test_users, test_history,test_history2,rr, test_labels

    def predict(self,epoch = '01'):
        test_items, test_users, test_history,test_history2,rr, test_labels = self.create_test_data('test')
        valid_items, valid_users, valid_history,valid_history2 ,rr1,valid_labels = self.create_test_data('valid')
        user_valid_baskets_df = self.valid_baskets.groupby('user_id')['item_id'].apply(list).reset_index()
        user_valid_baskets_dict = dict(zip( user_valid_baskets_df['user_id'],user_valid_baskets_df['item_id']))

        epoch_recall = []
        for epoch in range(1,6):
            print('epoch',epoch)
            epoch_str = str(epoch)
            if epoch<10:
                epoch_str = '0'+str(epoch)

            self.model.load_weights(self.data_path+'_weights.'+epoch_str+'.hdf5')
            y_pred = self.model.predict([valid_items,valid_users,valid_history,valid_history2,rr1],batch_size = 5000)
            predictions = [round(value) for value in y_pred.flatten().tolist()]
            accuracy = accuracy_score(valid_labels, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            recall_scores = []
            for user in user_valid_baskets_dict:
                top_items = []
                if user in self.user_id_mapper:
                    user_id = self.user_id_mapper[user]
                    indices = np.argwhere(valid_users == user_id)
                    item_scores = y_pred[indices].flatten()
                    item_ids = valid_items[indices].flatten()

                    item_score_dic = {}
                    for i, item_id in enumerate(item_ids):
                        item_score_dic[self.id_item_mapper[item_id]] = item_scores[i]
                    sorted_item_scores = sorted(item_score_dic.items(), key= lambda x: x[1], reverse = True)
                    top_items = [x[0] for x in sorted_item_scores]
                recall_scores.append(recall_k(user_valid_baskets_dict[user],top_items,
                                              len(user_valid_baskets_dict[user])))
            epoch_recall.append(np.mean(recall_scores))
        print(epoch_recall)
        print(np.argmax(np.array(epoch_recall)))
        best_epoch = np.argmax(np.array(epoch_recall)) + 1
        epoch_str = str(best_epoch)
        if best_epoch<10:
            epoch_str = '0'+str(best_epoch)
        print('best model:',self.data_path+'_weights.'+epoch_str+'.hdf5')
        print('best recall on valid:',epoch_recall[best_epoch-1])
        self.model.load_weights(self.data_path+'_weights.'+epoch_str+'.hdf5')
        y_pred = self.model.predict([test_items,test_users,test_history,test_history2,rr],batch_size = 5000)
        prediction_baskets = {}
        prediction_scores = {}
        for user in self.test_users:
            top_items = []
            if user in self.user_id_mapper:
                user_id = self.user_id_mapper[user]
                indices = np.argwhere(test_users == user_id)
                item_scores = y_pred[indices].flatten()
                item_ids = test_items[indices].flatten()
                item_score_dic = {}
                for i, item_id in enumerate(item_ids):
                    item_score_dic[self.id_item_mapper[item_id]] = item_scores[i]
                sorted_item_scores = sorted(item_score_dic.items(), key= lambda x: x[1], reverse = True)
                top_items = [x[0] for x in sorted_item_scores]
                prediction_scores[user] = sorted_item_scores
            
            prediction_baskets[user] = top_items

        return prediction_baskets
