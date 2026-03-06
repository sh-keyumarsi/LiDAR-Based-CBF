
class dataLogger():
    def __init__(self, max_data_num = 10000): # by default store 
        self.__data_len = max_data_num
        self.__stored_data = { 'time':[None]*max_data_num }
        self.__cur_idx = 0

        # only for plotting purposes
        self.__axes_list = {}


    def time_stamp(self, t): # should be called last after storing all new data
        self.__stored_data['time'][self.__cur_idx] = t
        self.__cur_idx += 1 # for next data filling

    def store_dictionary(self, dict):
        assert self.__cur_idx < self.__data_len, f"Data log exceeds max data_len: {self.__data_len}"
        for key, value in dict.items():
            if not (key in self.__stored_data): # assign default None array
                self.__stored_data[key] = [None]*self.__data_len          
            # Fill the array starting this index
            self.__stored_data[key][self.__cur_idx] = value

    def get_all_data(self): return self.__stored_data, self.__cur_idx
    def get_data_from_label(self, label): # Return up until recent data
        return self.__stored_data[label][:self.__cur_idx]
    
    def save_to_pkl(self, path):
        import pickle
        print('Storing the data to into: '+path, flush=True)
        with open(path, 'wb') as f:
            pickle.dump(dict(stored_data=self.__stored_data, last_idx=self.__cur_idx-1), f)
        print('Done.')


    # Plot logged data
    def plot_time_series_batch(self, ax, pre_string):
        # initialize plot to store the plot pointer
        dict_data = {'ax':ax, 'pl':{}}
        # plot all key with matching pre_string
        matches = [key for key in self.__stored_data if key.startswith(pre_string)]            
        for key in matches:
            dict_data['pl'][key], = dict_data['ax'].plot(0, 0, label=key.removeprefix(pre_string))
        # set grid and legend
        dict_data['ax'].grid(True)
        dict_data['ax'].set(xlabel="t [s]", ylabel=pre_string)
        # store data for update later
        self.__axes_list[pre_string] = dict_data

    def update_time_series_batch(self, pre_string, data_minmax = None):
        # data_minmax should be a tuple with 2 inputs
        if pre_string in self.__axes_list:
            dict_data = self.__axes_list[pre_string]
            # compute the time data
            min_idx, max_idx = 0, self.__cur_idx
            if data_minmax is not None: 
                min_idx, max_idx = data_minmax[0], data_minmax[1]
            time_data = self.__stored_data['time'][min_idx:max_idx]
            # check all matching keystring
            matches = [key for key in self.__stored_data if key.startswith(pre_string)]
            is_new_plot_added = False
            data_min, data_max = 0., 0.
            for key in matches:
                if key in dict_data['pl']:
                    key_data = self.__stored_data[key][min_idx:max_idx]
                    dict_data['pl'][key].set_data( time_data, key_data )
                else: # new data, make new plot
                    key_data = self.__stored_data[key][min_idx:max_idx]
                    dict_data['pl'][key], = dict_data['ax'].plot( time_data, key_data,
                        label=key.strip(pre_string))
                    is_new_plot_added = True
                # update min max for plotting
                data_min = min( data_min, min(key_data) ) 
                data_max = max( data_max, max(key_data) ) 
            # adjust time window
            dict_data['ax'].set(xlim= (time_data[0]-0.1, time_data[-1]+0.1), 
                ylim= (data_min-0.1, data_max+0.1))
            dict_data['ax'].legend(loc= 'best', prop={'size': 6})
            # update dictionary if needed
            if is_new_plot_added: self.__axes_list[pre_string] = dict_data
    