import numpy as np
import cv2
import os
from pathlib import Path
from mungling import DataMungling
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import timedelta

def extract_data_paths(recording_location:Path,subject:str):
    video=Path(str(recording_location)+'_labeled.mp4')
    data_csv=Path(str(recording_location)+'.csv')
    data_paths={'csv':data_csv,
                'video':video}
    return data_paths


class DrawLineWidget(object):
    def __init__(self,original_image):
        self.original_image = original_image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone) 

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    draw_line_widget = DrawLineWidget()
    while True:
        cv2.imshow('image', draw_line_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)

class Gesture(DataMungling):
    
    def __init__(self,data_paths:dict,name:str,pixels_cm:int)->None:
        self.name=name
        self.data_paths=data_paths
        self.pixels_cm=pixels_cm
        self.load_data()
        self.load_video()
        self.n_frames=self.get_frame_count(self.video)

        self.rename_dataframe()
        self.data=self.df_drop_first_rows(self.data,n_rows=2)
        self.data=self.data.iloc[0:self.n_frames] # clean dlc csv by using the number of frames video.
        # sometimes the df has one or two lines of ceros at the end
        self.data=self.data.astype(float) #change data types of df to float

    def load_video(self):
        self.video=cv2.VideoCapture(str(self.data_paths['video']))
    
    def load_data(self):
        self.data=pd.read_csv(self.data_paths['csv'])
    
    def rename_dataframe(self):
        """Join two first rows of deep lab cut dataframe, to rename dataframe columns
        """
        col_names=self.data.iloc[0,:].values+'_'+self.data.iloc[1,:].values
        self.data.columns=col_names

    def frames_to_matrix(self):

        ret=True
        self.frames=[]
        self.time=[]
        counter=0
        while ret==True:
            ret, frame = self.video.read()
            if ret==False:
                break
            counter=counter+1
            timestamp = np.round(self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,4)  # Convert to seconds
            self.frames.append(frame)
            self.time.append(timestamp)    
        self.time=np.array(self.time)
        self.frames=np.array(self.frames)
    
    def check_data_length(self):
        if len(self.frames)!=len(self.data):
          raise('Discrepancy frames and data length')
    
    def downsampling(self):
            # downsample if necessary
        if self.video_hz!=self.to_hz:
            factor=self.to_hz/self.video_hz
            n_rows=int(len(self.data)*factor)
            
            selected_rows=np.linspace(0,len(self.frames)-1,n_rows,dtype=int)
            self.frames=self.frames[selected_rows]
            self.data=self.data.iloc[selected_rows,:]
            self.data['diff_time']=1/self.to_hz
            self.time=self.time[selected_rows]
            final_hz=n_rows/(self.data['diff_time'].sum())
            self.data['frame_rate_final']=final_hz

            print(f'final_hz:{final_hz}')
        else:
            print('maintaing frame rate')
        self.check_data_length()
        
    def add_head_midpoint(self):
        self.data['head_x']=(self.data['LeftEar_x']+self.data['RightEar_x'])/2
        self.data['head_y']=(self.data['LeftEar_y']+self.data['RightEar_y'])/2

    def get_vectors(self):
        self.data['x_axis_vector']=self.data['nose_x']-self.data['head_x']
        self.data['y_axis_vector']=self.data['nose_y']-self.data['head_y']
        self.data['x_cricket_vector']=self.data['Cricket_x']-self.data['head_x']
        self.data['y_cricket_vector']=self.data['Cricket_y']-self.data['head_y']

    def orientation_head_cricket(self):
        angle1=np.arctan2(self.data['y_axis_vector'],self.data['x_axis_vector'])
        angle2 = np.arctan2(self.data['y_cricket_vector'],self.data['x_cricket_vector'])
        angle_between_lines = angle2 - angle1
        # Adjust the angle to be in the range of -pi to pi
        angle_between_lines = (angle_between_lines + np.pi) % (2 * np.pi) - np.pi
        # Convert the angle from radians to degrees
        angle_between_lines_degrees = np.degrees(angle_between_lines)

        self.data['angle_head_cricket']=angle_between_lines
        self.data['deg_head_cricket']=angle_between_lines_degrees


    def add_time(self):
        self.data['time']=self.time
        self.data['frame_num']=np.linspace(1,len(self.time),len(self.time),dtype=int)
    
    def add_distances(self):

        self.data['distance_px']=np.insert(self.distance_point(x=self.data['head_x'],
                                                y=self.data['head_y']),0,0)
        
        self.data['body_distance_px']=np.insert(self.distance_point(x=self.data['Body_x'],
                                                y=self.data['Body_y']),0,0)

        
        self.data['distance_head_cricket_px']=(self.distance_x_y(x1=self.data['head_x'],x2=self.data['Cricket_x'],
                                                y1=self.data['head_y'],y2=self.data['Cricket_y']))

        self.data['diff_head_cricket_px']=np.insert(np.diff(self.data['distance_head_cricket_px']),0,0)
        
        self.data['distance_cm']=self.data['distance_px']/self.pixels_cm

        self.data['body_distance_cm']=self.data['body_distance_px']/self.pixels_cm
        
        self.data['distance_head_cricket_cm']=self.data['distance_head_cricket_px']/self.pixels_cm
        
        self.data['diff_head_cricket_cm']=self.data['diff_head_cricket_px']/self.pixels_cm
    
    def add_temporal_data(self):

        diff_time=np.diff(self.data['time'],prepend=0)
        self.data['diff_time']=diff_time
        diff_gradient=np.diff(self.data['distance_head_cricket_cm'],prepend=0)
        self.data['gradient']=diff_gradient

        self.data.insert(loc=self.data.shape[1],
                    column='speed_cm_s',
                    value=self.data['distance_cm']/diff_time)
        
        self.data.insert(loc=self.data.shape[1],
                         column='body_speed_cm_s',
                         value=self.data['body_distance_cm']/diff_time)

        self.data.insert(loc=self.data.shape[1],
                    column='to_cricket_cm_s',
                    value=self.data['diff_head_cricket_cm']/diff_time)
        
        ## Acceleartion to relative to cricket
        diff_speed_head_cricket=np.diff(self.data['to_cricket_cm_s'],prepend=0)
        
        self.data.insert(loc=self.data.shape[1],
                    column='to_cricket_cm_s_2',
                    value=diff_speed_head_cricket/diff_time)
        
        ## Acceleration relative to head
        diff_speed_head=np.diff(self.data['speed_cm_s'],prepend=0)
        self.data.insert(loc=self.data.shape[1],
            column='speed_cm_s_2',
            value=diff_speed_head/diff_time)

        ## Acceleration relative to body
        diff_speed_body=np.diff(self.data['body_speed_cm_s'],prepend=0)
        self.data.insert(loc=self.data.shape[1],
            column='body_speed_cm_s_2',
            value=diff_speed_body/diff_time)

    def categorize_behaviours(self,approach:str,contact:str):
        self.approach_filter_string=approach
        self.contact_filter_string=contact
        approach_df=self.data.query(approach)
        contact_df=self.data.query(contact)

        exploration_index=(~self.data.index.isin(contact_df.index )) & (~self.data.index.isin(approach_df.index ))
        exploration_df=self.data[exploration_index]

        approach_df['behaviour']='approach'
        contact_df['behaviour']='contact'
        exploration_df['behaviour']='exploration'

        self.behaviour={'approach':approach_df,
                        'contact':contact_df,
                        'exploration':exploration_df}
        print('Categorize behaviours in approaches, contact and exploration')

        index_contact=self.behaviour['contact'].index
        index_exploration=self.behaviour['exploration'].index
        index_approach=self.behaviour['approach'].index

        index_1=index_approach.intersection(index_contact)
        index_2=index_approach.union(index_contact)

        if any(index_1) | any(index_2.intersection(index_exploration)):
            print('overlapping behaviours')
        else:
            print('append behaviours to main dataframe')
            self.data= self.behaviour['approach'].append(self.behaviour['exploration'],ignore_index=False).append(self.behaviour['contact'],ignore_index=False)
            self.data=self.data.sort_index()


    def view_data(self):
        
        for row in  self.data.iterrows():
            # plt.axis([150, 450,0, 250])
            
            plt.imshow(self.frames[row[0]])
            plt.plot(row[1]['Cricket_x'],row[1]['Cricket_y'],'.',color='orange' )
            skeleton_x=['nose_x','LeftEar_x','RightEar_x','Body_x','Tail_x']
            skeleton_y=['nose_y','LeftEar_y','RightEar_y','Body_y','Tail_y']

            # plt.plot(row[1]['nose_x'],row[1]['nose_y'],'.',color='blue' )
            # plt.plot(row[1]['LeftEar_x'],row[1]['LeftEar_y'],'.',color='red' )
            # plt.plot(row[1]['RightEar_x'],row[1]['RightEar_y'],'.',color='red' )
            # plt.plot(row[1]['Body_x'],row[1]['Body_y'],'.',color='blue' )
            # plt.plot(row[1]['Tail_x'],row[1]['Tail_y'],'.',color='black' )

            
            plt.plot(row[1][skeleton_x],row[1][skeleton_y],color='blue')
            plt.plot(row[1]['head_x'],row[1]['head_y'],'.',color='red')
            plt.text(row[1]['head_x'],row[1]['head_y'],np.round(row[1]['deg_head_cricket'],4),color='red')
            plt.text(row[1]['head_x'],row[1]['head_y']+30,row[1]['behaviour'],color='blue')

            plt.pause(0.05)
            plt.clf()  # Clear the figure for the next framek

            plt.show()

    def compute_behaviour_matrix(self,behaviour_thr:float=0.2):
        """Computes a habiour matrix that separates and meassures the duration
        of each type of behaviour through the the dataframe.data
        It first uses the method add_beahvior_counter, to clasiffy the 3 tyes of behaviour through time
        It then groups the data and calculates the values

        Args:
            behaviour_thr (float): time in seconds use as thershold for the minimum duration of a 
        type of behaviour
        """
        self.behaviour_thr=behaviour_thr  #save parameters
        # self.add_behaviour_counter()
        self.behaviour_matrix=(self.data.groupby(['beh_counter','behaviour'])
                .agg(diff_time=('diff_time',np.sum),
                     time_0=('time',np.min),
                     time_end=('time',np.max),
                     max_speed=('speed_cm_s',np.max),
                     min_speed=('speed_cm_s',np.min),
                     mean_speed=('speed_cm_s',np.mean))).reset_index()
                    #  .query('diff_time>@behaviour_thr')).reset_index()

    def add_behaviour_counter(self): 

        """This methods adds a new column to the dataframe that gives a new
        index everytime there is a behaviour change
        """
        counter=0
        beh=self.data['behaviour'].values
        beh_counter=np.int_(np.zeros(len(beh)))
        for row, behaviour in enumerate(beh[:-2]):

            if beh_counter[row]==counter and row>0:
                continue
            if behaviour=='nada':
                beh_counter[row]=0
                continue

            if any((beh[row:row+2]==behaviour)==True):
                counter=counter+1
                try:
                    behaviour_change_index=np.where(beh[row::]!=behaviour)[0][0]+row
                    beh_counter[row:behaviour_change_index]=counter

                except IndexError:
                    beh_counter[row:]=counter

        self.data['beh_counter']=beh_counter


    def add_behaviour_counter_refactor(self):
        beh=self.data['behaviour']

        # Create a mask for changes in the string and exclude nan vlaues from changes
        # mask = (beh != beh.shift()) & ~(beh.isna()) & ~(beh.shift(-1).isna())
        # mask = (beh != beh.shift()) & ~(beh.isna()) 
        # mask=mask = (beh != beh.shift()) & ~(beh.shift().isna())
        # data_1['beh_counter']=0

        data_aux=self.data['behaviour'].to_frame().dropna() 
        #compute behaviour counter without nans

        mask = (data_aux != data_aux.shift()) 
        data_aux['beh_counter']=mask.cumsum() # Add behvoir 
        self.data['beh_counter']=data_aux['beh_counter']


        # self.data['beh_counter']=mask.cumsum() #now cumulutive sunm of tru values

    def plot_behaviours(self,y_index:int,fig:matplotlib,ax:matplotlib.axis):
        y=np.ones(len(self.behaviour_matrix))

        def behaviour_color():
            if behaviour=='contact':
                col='r'
            elif behaviour=='exploration':
                col='g'
            elif behaviour=='approach':
                col='b'
            return col

        for behaviour in pd.unique(self.behaviour_matrix['behaviour']):
            for row, value in self.behaviour_matrix[self.behaviour_matrix['behaviour']==behaviour].iterrows():
                time=np.linspace(value['time_0'],value['time_end'],int(value['diff_time']*60))
                ax.plot(time,np.ones(len(time))*y_index+1,color=behaviour_color(),linewidth=3,scaley=True)

        ax.set_xlabel('time(s)')
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        return fig, ax

    def save_processed_video(self,video_hz:float,video_name:str):
        """Method to save the proccesssed video with frame annotations of interest
        such as frame count, orientaton, behaviour type. 

        Args:
            video_hz (float): fps 
            video_name (str): <videoname>.avi !!IMP, introduce the format name
        """
        # Create a sample NumPy array representing a video
        # Replace this with your own video data
        # Sample data: 100 frames of 100x100 random images
        video_array =self.frames
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Choose the codec (XVID for .avi)
        out = cv2.VideoWriter(video_name, fourcc,video_hz, (self.frames.shape[2], self.frames.shape[1]))  # Parameters: filename, codec, fps, frame_size

        behaviour_col_index=np.where(self.data.columns=='behaviour')[0][0]
        orientation_col_index=np.where(self.data.columns=='deg_head_cricket')[0][0]
        time_col_index=np.where(self.data.columns=='time')[0][0]
        head_x_index=np.where(self.data.columns=='head_x')[0][0]
        head_y_index=np.where(self.data.columns=='head_y')[0][0]
        cricket_x_index=np.where(self.data.columns=='Cricket_x')[0][0]
        cricket_y_index=np.where(self.data.columns=='Cricket_y')[0][0]
        speed_index=np.where(self.data.columns=='speed_cm_s')[0][0]
        body_speed_index=np.where(self.data.columns=='body_speed_cm_s')[0][0]
        distance_index=np.where(self.data.columns=='distance_cm')[0][0]
        # Iterate over each frame in the video array
        for index,frame in enumerate(video_array):
            # # Add text to the frame
            text = f'frame: {index}'  # Example text
            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            time=self.data.iat[index,time_col_index]
            text = f'time: {str(timedelta(seconds=time))}'  # Example text
            cv2.putText(frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            behaviour=self.data.iat[index,behaviour_col_index]
            text=f'behaviour: {behaviour}'
            cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            orientation=np.round(self.data.iat[index,orientation_col_index],2)
            text=f'orientation: {orientation}'
            cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            speed=np.round(self.data.iat[index,speed_index],2)
            text=f'head speed cm/s: {speed}'
            cv2.putText(frame, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            body_speed=np.round(self.data.iat[index,body_speed_index],2)
            text=f'body speed cm/s: {body_speed}'
            cv2.putText(frame, text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            distance=np.round(self.data.iat[index,distance_index],2)
            text=f'distance cm: {distance}'
            cv2.putText(frame, text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            head_x=int(self.data.iat[index,head_x_index])
            head_y=int(self.data.iat[index,head_y_index])

            cv2.circle(frame, (head_x, head_y), radius=2, color=(0, 255, 0), thickness=-1)

            cricket_x=int(self.data.iat[index,cricket_x_index])
            cricket_y=int(self.data.iat[index,cricket_y_index])

            cv2.circle(frame, (cricket_x, cricket_y), radius=2, color=(255, 0, 0), thickness=-1)

            # Write the frame to the video file
            out.write(frame)



        # Release the VideoWriter object
        out.release()

        print("Video saved successfully!")


def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, align='edge',normalize=True, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    if normalize:
      angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins,density=True)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1],radius,zorder=1, align=align, width=widths, color='black',
           edgecolor='w', fill=True, linewidth=2,alpha=0.7)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        ax.set_xticklabels(label)

    else:
        ax.set_xticklabels(['0', '45', '90', '135', '180','-135', '-90', '-45', '0'])


    ax.set_theta_zero_location("N") 
    ax.set_theta_direction(-1)
    return ax


