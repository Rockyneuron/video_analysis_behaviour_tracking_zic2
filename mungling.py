import pandas as pd
import numpy as np
import cv2
class DataMungling:

    def __init__(self) -> None:
        pass

    def cut_dataframe_by_column_values(self,
                                    df:pd.DataFrame=pd.DataFrame,
                                    initial_label:str='asset',
                                    final_label:str='end_of_experiment',
                                    filter_column:str='label'):
        
        """Filter any dataframe by two column values. Cut a dataframe by two 
        reference values:
        1) An intial value by first order of appearance in the df
        2) A final value by first order of appearance in the df

        Args:
            df (pd.DataFrame): _description_
            inital_label (str, optional): name of the first label. Defaults to 'asset'.
            final_label (str, optional): name of the final label. Defaults to 'end_of_experiment'.
            filter_column (str, optional): name of the column used for filtering the df. Defaults to 'label'.

        Returns:
            pd.Dataframe: _description_
        """

        index_initial=df.loc[df[filter_column]==initial_label].index[0]
        index_final=df.loc[df[filter_column]==final_label]
        if index_final.empty:
            df_final=df.loc[index_initial::,:]
        else:
            df_final=df.loc[index_initial:index_final.index.values[0],:]
        return df_final

    def filter_series_string(self,df:pd.Series,label:str='label'):
        return df.str.contains(label,na=False) 
    
    def filter_series_list_string(self,df:pd.Series,label:list[str]):
        """Function to filfer a pd.Series by a common list of strings of 
        coincidences

        Args:
            df (pd.Series): _description_
            label (list[str]): list of strings

        Returns:
            _df_: filtered pandas series datrame
            _bool_: boolean index vector  
        """
        for n,name in enumerate(label):
            if n==0:
                index=self.filter_series_string(df,name)
            else:
                index=self.filter_series_string(df,name) | index

        return (df[index], index)
    

    def calculate_contrast(self, x,y):
        """Function to calculate michealson contrast
        Args:
            x (_np.array_): _description_
            y (_np.arry_): _description_
        """
        contrast=(x-y)/(x+y)
        return contrast

    def refactor_df_to_categorical(self,df:pd.DataFrame,col_names:list[str]): 
        """This function refactors any 2d matrix dataframe to a categorical dataframe
        It takes the values of the index and the values of the colums as catetorical 
        variables and the final value as a continous variable.

        Args:
            df (pd.DataFrame): _description_
            col_names (list[str]): [index_column_name,column_var_name,variable name]

        Returns:
            _type_: a 3d categorical matrix with the columns ordered as in col_names.
            col_names (list[str]): [index_column_name,column_var_name,variable name]
        """

        series_list=[]
        df.reset_index(inplace=True)
        df=df.rename(columns={'index':col_names[0]})

        for row in df.iterrows():
            df_aux=pd.DataFrame(row[1][1:]) #remove the index name from the series
            print(col_names[0])
            print(row[1][0])
            df_aux[col_names[1]]=row[1][0]      #add the value of the index as another column
            df_aux.reset_index(inplace=True) #remove index
            df_aux.columns=[col_names[1],col_names[2],col_names[0]] #rename columns
            series_list.append(df_aux)
            
        df_final=pd.concat(series_list)
        df_final=df_final[col_names] #rearragne column names
        return df_final
    
    def distance_point(self,x:pd.Series,y:pd.Series):
        """function to calculate the distance between a single point
        Args:
            x (pd.Series): array of x values 
            y (pd.Series): array of y values
        Returns:
            _type_: _description_
        """
        return np.sqrt((np.diff(x)**2)+((np.diff(y))**2))


    def distance_x_y(self,x1:pd.Series,x2:pd.Series,y1:pd.Series,y2:pd.Series):
        """function to calculate the distance between two points
        Args:
            x (pd.Series): array of x values 
            y (pd.Series): array of y values
        Returns:
            _type_: _description_
        """
        return np.sqrt(((x2-x1)**2)+((y2-y1)**2))

    def calculate_contrast(x,y):
        """Function to calculate michealson contrast
        Args:
            x (_np.array_): _description_
            y (_np.arry_): _description_
        """
        contrast=(x-y)/(x+y)
        # print(contrast)
        return contrast

    def df_drop_first_rows(self,data:pd.DataFrame,n_rows:int,reset_index=True):
        """Drops firs n rows of a dataframe

        Args:
            n_rows (int): number of rows to drop
            reset_index (bool, optional): reset index of dataframe. Defaults to True.
        """
        data=data.iloc[n_rows:,:] 
        if reset_index:
            data.reset_index(drop=True,inplace=True)
        return data
    
    def df_drop_final_rows(self,data:pd.DataFrame,n_rows:int,reset_index=True):
        """Drops final n rows of a dataframe

        Args:
            n_rows (int): number of rows to drop
            reset_index (bool, optional): reset index of dataframe. Defaults to True.
        """        
        data=data.iloc[:-n_rows] # drop last 2 rwos
        if reset_index:
            data.reset_index(drop=True,inplace=True)
        return data
    
    def get_frame_count(self,video:cv2):
        """Get frame count from CV.video

        Args:
            video (cv2.video): _description_

        Returns:
            _type_: _description_
        """
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def filter_rows_by_temporal_values(dataframe:pd.DataFrame,time_column:str,ini_value:float,end_value:float):
        """_summary_

        Args:
            dataframe (pd.DataFrame): dataframe of interest 
            time_column (str): column to use for filtering
            ini_value (float): initial temporal value
            end_value (float): final temporal value

        Returns:
            _type_: segmented dataframe
        """
        segmented_df=dataframe[
            (dataframe[time_column]>=ini_value)&
            (dataframe[time_column]< end_value)
        ]
        return segmented_df
    
def data_summary(df:pd.DataFrame,col_name:str,group_by:list):
    """Funtion for summarrinz data

    Args:
        df (pd.DataFrame): _description_
        col_name (str): bname of the column to summarize
        group_by (str): groupby category
    """
    summary= (df.groupby(group_by)
                .agg(mean=(col_name,np.mean),
                    std=(col_name,np.std),
                    q1= (col_name,lambda x: x.quantile(0.25) ),
                    q2= (col_name,lambda x: x.quantile(0.50) ),
                    median= (col_name,lambda x: x.median()),
                    q3=(col_name,lambda x: x.quantile(0.75)),
                    IQR=(col_name,lambda x:x.quantile(0.75)- x.quantile(0.25)),
                    max=(col_name,np.max),
                    min=(col_name,np.min)))
    
    summary['low_bound_tukey']=summary['q1']-3*summary['IQR']
    summary['upper_bound_tukey']=summary['q3']+3*summary['IQR']

    display(summary)
    return summary