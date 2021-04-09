# OpenCV program to perform Edge detection in real time 
# import libraries of python OpenCV  
import cv2 #Importing Open CV4 into your Python Code
import os
from os import listdir,makedirs
from os.path import isfile,join
from PIL import Image
import numpy as np #Importing Numpy for array manipulation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import csv
from time import process_time


def Vid_OG():
    # capture frames from a camera
    cap = cv2.VideoCapture(0) 
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter('images/output_live_5.avi', fourcc, 20.0, (640,480))

    #first_frame=mpimg.imread('New_Frame.jpg')
    #cv2.imshow('f', first_frame)

    #elapsed_time = 0    
    while(True):        
        # reads frames from a camera 
        ret, frame = cap.read()

        # converting BGR to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
          
        # define range of red color in HSV 
        lower_red = np.array([30,150,50]) 
        upper_red = np.array([255,255,180]) 
          
        # create a red HSV colour boundary and  
        # threshold HSV image 
        mask = cv2.inRange(hsv, lower_red, upper_red) 

        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(frame,frame, mask= mask) 
        cv2.imshow('Original',frame)
        out.write(frame)
        # Wait for Esc key to stop 
        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def Edge_to_frames():
    cap = cv2.VideoCapture('images/output_live_5.avi')
    
    # reads frames from a camera 
    ret, frame = cap.read()
    
    i=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        edges = cv2.Canny(frame,100,200)
        if ret == False :
            print('not working')
            break
        if i%5 == 0:
            cv2.imwrite('output_Edge50/select_output'+str(i)+'.jpg',edges)# creating folder for data
            
        i+=1
    
    cap.release()
    
    cv2.destroyAllWindows()

def Mean_Shift():
    elapsed_time = 0
    #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    cap = cv2.VideoCapture('images/output_live_5.avi')
    #out2 = cv2.VideoWriter('images2/output2_live_5.avi',fourcc, 5.0, (640,480))
    # reads frames from a camera 
    ret, frame = cap.read()

    
    # setup default location of window
    y, h, x, w = 300, 100, 300, 100
    track_window = (x, y, w, h)

    # Crop region of interest for tracking
    roi = frame[y:y+h, x:x+w]
    
    # Convert cropped window to HSV color space
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Obtain the color histogram of the ROI
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])
    # Normalize values to lie between the range 0, 255
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            
    # Setup the termination criteria
    #We stop calculating the centroid shift after ten iterations
    #or if the centroid has moved at least 1 pixel
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    i = 0
    while i<50:
        #read webcame frame
        ret, frame = cap.read()
    
        if ret == True:
            tic1 = process_time()
            #convert to hsv
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #calculate histogram back projection
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            #apply meanshift to get new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            #draw it on image
            x, y, w, h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

            #cv2.imwrite('images2/output2_live_5.avi',img2)
            # put dot and highlight the center
            cv2.circle(img2, (int(x+w/2), int(y+h/2)), 5, (255, 255, 255), -1)
            cent_MS = int(x+w/2), int(y+h/2)
            cent_MS_x, cent_MS_y = cent_MS
            cv2.imwrite('output_MS50/select_output'+str(i)+'.jpg',img2)

            toc1 = process_time()
            MS_elapsed_time = toc1-tic1
            out_file = open("Centroid_+_Timer_MS.csv", "a") # open the file for the data to get written to
            writer = csv.writer(out_file) # opens csv writer
            writer.writerow([cent_MS_x, cent_MS_y, MS_elapsed_time]) # writes to the csv
            out_file.close() #closes the file
            

            if i <10:
                ###STEP1### Run this code first to get your PMF Histograms 
                #then comment it out and run it again for Step2.
                #heights,bins = np.histogram(img2,bins=256)
                #heights = heights/sum(heights)
                #plt.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="blue", alpha=0.5)
                #plt.savefig('PMF_MS_HISTO/select_output'+str(i)+'.jpg')
                
                ###Step2### Uncomment this part #! and run code again to get your CDF Plots.
                data = cv2.imread('PMF_MS_HISTO/select_output'+str(i)+'.jpg')
                # Choose how many bins you want here
                num_bins = 256

                # Use the histogram function to bin the data
                counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)

                # Now find the cdf
                cdf = np.cumsum(counts)

                # And finally plot the cdf
                plt.plot(bin_edges[1:], cdf)
                plt.savefig('CDF_MS/select_output'+str(i)+'.jpg')

                

        else:
            break
        i = i+1
        ############### RERUN CODE TO GET CAMSHIFT##############

def CamShift():
    elapsed_time = 0
    #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    cap = cv2.VideoCapture('images/output_live_5.avi')
    #out2 = cv2.VideoWriter('images2/output2_live_5.avi',fourcc, 5.0, (640,480))
    # reads frames from a camera 
    ret, frame = cap.read()

    
    # setup default location of window
    y, h, x, w = 300, 100, 300, 100
    track_window = (x, y, w, h)
    # Crop region of interest for tracking
    roi = frame[y:y+h, x:x+w]

    # Convert cropped window to HSV color space
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Create a mask between the HSV bounds
    lower_purple = np.array([120,0,0])
    upper_purple = np.array([175,255,255])
    mask = cv2.inRange(hsv_roi, lower_purple, upper_purple)
    
    
    # Obtain the color histogram of the ROI
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

    # Normalize values to lie between the range 0, 255
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    

    # Setup the termination criteria
    # We stop calculating the centroid shift after ten iterations
    # or if the centroid has moved at least 1 pixel
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    i = 0
    while i<50:
        #read webcame frame
        ret, frame = cap.read()

        if ret == True:
            tic2 = process_time()
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Calculate the histogram back projection
            
            # Each pixel's value is it's probability
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            
            # apply Camshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window,term_crit)
            
            # Draw it on image
            # We use polylines to represent Adaptive box
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img3 = cv2.polylines(frame,[pts],True, (0, 255, 0),2)
            arr = np.asarray(pts)
            length = arr.shape[0]
            sum_x = np.sum(arr[:, 0])
            sum_y = np.sum(arr[:, 1])
            #return sum_x / length, sum_y / length
            cv2.circle(img3, (int(sum_x / length),int(sum_y / length)), 5, (255, 255, 255), -1)
            #cv2.circle(img3, (int([pts])), 5, (255, 255, 255), -1)
            cent_CAM = (int(sum_x / length),int(sum_y / length))
            cent_CAM_x, cent_CAM_y = cent_CAM
            cv2.imwrite('output_Cam50/select_output'+str(i)+'.jpg',img3)
           
            toc2 = process_time()
            elapsed_time = toc2-tic2
            out_file = open("Centroid_+_Timer_CAM.csv", "a") # open the file for the data to get written to
            writer = csv.writer(out_file) # opens csv writer
            writer.writerow([cent_CAM_x, cent_CAM_y, elapsed_time]) # writes to the csv
            out_file.close() #closes the file

            if i <10:
                ###STEP1### Run this code first to get your PMF Histograms 
                #then comment it out and run it again for Step2.
                #heights,bins = np.histogram(img3,bins=256)
                #heights = heights/sum(heights)
                #plt.bar(bins[:-1],heights,width=(max(bins) - min(bins))/len(bins), color="blue", alpha=0.5)
                #plt.savefig('PMF_CAM_HISTO/select_output'+str(i)+'.jpg')
                
                ###Step2### Uncomment this part #! and run code again to get your CDF Plots.
                data = cv2.imread('PMF_CAM_HISTO/select_output'+str(i)+'.jpg')
                # Choose how many bins you want here
                num_bins = 256

                # Use the histogram function to bin the data
                counts, bin_edges = np.histogram(data, bins=num_bins, normed=True)

                # Now find the cdf
                cdf = np.cumsum(counts)

                # And finally plot the cdf
                plt.plot(bin_edges[1:], cdf)
                plt.savefig('CDF_CAM/select_output'+str(i)+'.jpg')


        else:
            break
        i = i+1

        ############### RERUN CODE TO GET MSCentroid ##############

def MSCentroid():
    x = []
    y = []
    z = []

    #data = pd.read_csv('Centroid_+_Timer_MS.csv') # importing the original csv file
    # Load the CSV file into a panda dataframe
    with open('Centroid_+_Timer_MS.csv') as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=',')
        for x1,y1,z1 in csv_reader:
            #print(x1)
            x.append(int(x1))
            #print(y1)
            y.append(int(y1))
            z.append(int(float(z1)))
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional line
        zline = z
        xline = x
        yline = y
        ax.plot3D(xline, yline, zline, 'gray')

        # Data for three-dimensional scattered points
        ax.scatter3D(x, y, z,  cmap='Greens')
        plt.savefig('CDF_MS/Centroid.jpg')
        plt.show()

        ############### RERUN CODE TO GET CAMCentroid ##############

def CamCentroid():
    x = []
    y = []
    z = []

    #data = pd.read_csv('Centroid_+_Timer_MS.csv') # importing the original csv file
    # Load the CSV file into a panda dataframe
    with open('Centroid_+_Timer_CAM.csv') as csvfile:
        csv_reader = csv.reader(csvfile,delimiter=',')
        for x1,y1,z1 in csv_reader:
            #print(x1)
            x.append(int(x1))
            #print(y1)
            y.append(int(y1))
            z.append(int(float(z1)))
        ax = plt.axes(projection='3d')
        # Data for a three-dimensional line
        zline = z
        xline = x
        yline = y
        ax.plot3D(xline, yline, zline, 'gray')

        # Data for three-dimensional scattered points
        ax.scatter3D(x, y, z,  cmap='Greens')
        plt.savefig('CDF_CAM/Centroid.jpg')
        plt.show()        


if __name__ == '__main__':
    os.system('clear')
    select = input('This is a Meanshift and Camshift program using your recorded video:\nEnter 1 to record new video then press Enter:\nEnter 2 to the run the code on previously recorded video then press Enter:\nEnter 3 to run the Centroids code for MS and CAM then press Enter:\nNote you will have to run these steps in Numeric Order to get all the needed info and graphs thanks: ')
    if select == '1':
        os.system('clear')
        select1 = input('Would you lik to record a new video yes/no:\n enter y if yes or enter n to exit and run code:\n if you record new video press escape to stop:\n')
        if select1 == 'y':
            Vid_OG()
            
        elif select1 == 'n':
            exit()
        else:
            print('Not a valid selection')
            exit()

    if select == '2':
        os.system('clear')
        select2 = input('Enter 1 to run Meanshift, Enter 2 to run CamShift, and then press the Enter key:\n')
        if select2 == '1':
            os.system('clear')
            print('Running the MeanShift please wait:\n') 
            Mean_Shift()
            Edge_to_frames()
            exit()

        elif select2 == '2':
            os.system('clear')
            print('Running the CamShift please wait:\n')
            CamShift()
            exit()

    if select == '3':
        os.system('clear')
        select3 = input('Enter 1 to run Meanshift Centroid, Enter 2 to run CamShift Centroid, and then press the Enter key:\n')

        if select3 == '1':
            os.system('clear')
            print('Running the MeanShift Centroid please wait:\n')
            MSCentroid()
            exit()

        elif select3 == '2':
            os.system('clear')
            print('Running the CamShift Centroid please wait:\n')
            CamCentroid()
            exit()

    else:
        print('Not a valid selection:\n')
        exit()

