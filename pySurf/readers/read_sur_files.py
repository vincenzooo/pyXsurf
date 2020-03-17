import numpy as np
import pdb

def fread(fid, nelements, dtype):
    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype
    #pdb.set_trace()
    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = nelements #(nelements, 1)
     
    return data_array

"""
%read char array until first '\0'
function s=freadChar(fileID,bytes)
s=char(fread(fileID,bytes,'char')');
tmp= strfind((s),(char(0)));
if ~isempty(tmp)
    s=s(1:tmp(1));
end
%fix encoding
s=strrep(s,'','');
end
"""     

def freadChar(fileID,n):
    a= fread(fileID,n,np.uint8)
    s= bytes(a) 
    return "".join(map(chr,s)).strip('\x00')  #strip terminal nulls  
                    #this works where s.decode().strip('\x00') fails 
                    # e.g if first character is \xb5 (greek mu for micro)
    
class Res():
    pass
    
def readsur(filepath,raw=False):
    """
    reads surface data from Surf file format 2010. returns an object with all properties from the sur file.
    Data and axes are contained respectively in res.points, res.xAxis and res.yAxis.
    
    filepath is the full path to sur file.
    `raw` flag determines if points property contains internal representation rather than rescaled and 
    reshaped values (this uses half memory space wrt original matlab function).   
    
    returns an object with all properties from the sur file.
        res.points contains reshaped and scaled or as internally represented in the sur file 
            if `raw` flag is set True.
        res.x/yAxis contains the corresponding axis.
        
    ported to python by Vincenzo Cotroneo 2018/04/18 from matlab routine by Eike Foremny, v1.0 23.02.2017
        
    """
    fileID = open(filepath, 'rb')
    res=Res()

    res.signature           =   freadChar(fileID,12);               #1
    res.format              =   fread(fileID,1,'int16');            #2
    res.objNum              =   fread(fileID,1,'int16');            #3
    res.version             =   fread(fileID,1,'int16');            #4
    res.objType             =   fread(fileID,1,'int16');            #5
    res.objName             =   freadChar(fileID,30);               #6
    res.operatorName        =   freadChar(fileID,30);               #7

    res.materialCode        =   fread(fileID,1,'int16');            #8
    res.acquisitionType     =   fread(fileID,1,'int16');            #9
    res.rangeType           =   fread(fileID,1,'int16');            #10
    res.specialPoints       =   fread(fileID,1,'int16');            #11
    res.absoluteHeights     =   fread(fileID,1,'int16');            #12
    res.gaugeResolution     =   fread(fileID,1,'single'); #fread(fileID,1,'float');            #13

    freadChar(fileID,4);        #reserved space. for unknown reason this works in matlab, not in python  #14

    #data format of pints
    res.sizeOfPoints        =   fread(fileID,1,'int16');            #15

    res.zMin                =   fread(fileID,1,'int32');            #16
    res.zMax                =   fread(fileID,1,'int32');            #17

    #Number of points per axis
    res.xPoints             =   int(fread(fileID,1,'int32'))            #18
    res.yPoints             =   int(fread(fileID,1,'int32'))            #19

    res.totalNumberOfPoints =   fread(fileID,1,'int32');            #20
    
    #Distance between points
    res.xSpacing            =   fread(fileID,1,'single') #'float'); #21
    res.ySpacing            =   fread(fileID,1,'single') #'float'); #22
    res.zSpacing            =   fread(fileID,1,'single') #'float'); #23

    #Name of axis
    res.xName               =   freadChar(fileID,16);               #24
    res.yName               =   freadChar(fileID,16);               #25
    res.zName               =   freadChar(fileID,16);               #26

    #Unit of distance between points
    res.xStepUnit           =   freadChar(fileID,16);               #27
    res.yStepUnit           =   freadChar(fileID,16);               #28
    res.zStepUnit           =   freadChar(fileID,16);               #29

    #Unit of axis
    res.xLengthUnit         =   freadChar(fileID,16);               #30
    res.yLengthUnit         =   freadChar(fileID,16);               #31
    res.zLengthUnit         =   freadChar(fileID,16);               #32

    #Skaling of distance between points
    res.xUnitRatio          =   fread(fileID,1,'single') #'float'); #33
    res.yUnitRatio          =   fread(fileID,1,'single') #'float'); #34
    res.zUnitRatio          =   fread(fileID,1,'single') #'float'); #35

    res.imprint             =   fread(fileID,1,'int16') #'float'); #36
    res.inverted            =   fread(fileID,1,'int16') #'float'); #37
    res.levelled            =   fread(fileID,1,'int16') #'float'); #38
    freadChar(fileID,12);       #obsolete                           #39

    #timestamp
    res.startSeconds        =   fread(fileID,1,'int16');            #40
    res.startMinutes        =   fread(fileID,1,'int16');            #41
    res.startHours          =   fread(fileID,1,'int16');            #42
    res.startDays           =   fread(fileID,1,'int16');            #43
    res.startMonths         =   fread(fileID,1,'int16');            #44
    res.startYears          =   fread(fileID,1,'int16');            #45
    res.startWeekDay        =   fread(fileID,1,'int16');            #46
    res.measurementDuration =   fread(fileID,1,'single') #'float'); #47

    freadChar(fileID,10);       #obsolete                           #48

    #Size of comment field
    res.commentSize         =   fread(fileID,1,'int16');            #49
    res.privateSize         =   fread(fileID,1,'int16');            #50

    res.clientZone          =   freadChar(fileID,128);              #51

    #Axis offset
    res.xOffset             =   fread(fileID,1,'single') #'float'); #52
    res.yOffset             =   fread(fileID,1,'single') #'float'); #53
    res.zOffset             =   fread(fileID,1,'single') #'float'); #54

    #temperature scale
    res.tSpacing            =   fread(fileID,1,'single') #'float'); #55
    res.tOffset             =   fread(fileID,1,'single') #'float'); #56
    res.tStepUnit           =   freadChar(fileID,13);               #57
    res.tAxisName           =   freadChar(fileID,13);               #58

    res.comment             =   freadChar(fileID,int(res.commentSize));  #59
    res.private             =   freadChar(fileID,int(res.privateSize));  #60

    #read datapoints
    if res.sizeOfPoints == 16:                                       #61
        res.points          =   fread(fileID,-1,'int16')
    elif res.sizeOfPoints == 32:
        res.points          =   fread(fileID,-1,'int32')        
    else:
        raise ValueError("data lack property sizeOfPoints (or file is in the wrong format)")

    if not(raw):
        #reshape datapoints into 2D-Matrix
        res.points       =   np.reshape(res.points,[res.yPoints,res.xPoints])   #switched x-y 20180503
                    #it was giving interlaced output with profilometer data (never noticed on square CCI images).

    #Scale datapoints without Offset
    res.points       =   (res.points-res.zMin) * res.zSpacing/ res.zUnitRatio;

    #Generate Axis without Offset;
    res.xAxis               =   np.linspace(0,res.xSpacing * res.xPoints/ res.xUnitRatio, res.xPoints);
    res.yAxis               =   np.linspace(0,res.ySpacing * res.yPoints/ res.yUnitRatio, res.yPoints);

    fileID.close();
    return res
    
if __name__=="__main__":
    
    from pySurf.data2D import plot_data
    
    df=r'test\input_data\profilometer\04_test_directions\05_xysurf_pp_Height.sur'
    res=readsur(df)
    plt.figure()
    plot_data(res.points,res.xAxis,res.yAxis)
    
    
    