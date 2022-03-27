res=b_readSur('G:\My Drive\Shared by Vincenzo\Metrology logs and data\measure_data\2018_04_17\02_PCO1.2S03_xymarker_Height.sur');


function res=b_readSur(path)
%b_readSur(path) reads surface data from Surf file format 2010.
%   path = path to .sur file.
%   returns struct with all properties from the sur file.
%       struct.points contains all points as in the sur file.
%       struct.pointsAligned contains all points reshaped and scaled.
%       struct.x/yAxis contains the corresponding axis.
%
% Eike Foremny, 23.02.2017

fileID=fopen(path,'r','l');


res.signature           =   freadChar(fileID,12);               %1
res.format              =   fread(fileID,1,'int16');            %2
res.objNum              =   fread(fileID,1,'int16');            %3
res.version             =   fread(fileID,1,'int16');            %4
res.objType             =   fread(fileID,1,'int16');            %5
res.objName             =   freadChar(fileID,30);               %6
res.operatorName        =   freadChar(fileID,30);               %7


res.materialCode        =   fread(fileID,1,'int16');            %8
res.acquisitionType     =   fread(fileID,1,'int16');            %9
res.rangeType           =   fread(fileID,1,'int16');            %10
res.specialPoints       =   fread(fileID,1,'int16');            %11
res.absoluteHeights     =   fread(fileID,1,'int16');            %12
res.gaugeResulution     =   fread(fileID,1,'float');            %13

freadChar(fileID,4);        %reserved space                     %14

%data format of pints
res.sizeOfPoints        =   fread(fileID,1,'int16');            %15

res.zMin                =   fread(fileID,1,'int32');            %16
res.zMax                =   fread(fileID,1,'int32');            %17

%Number of points per axis
res.xPoints             =   fread(fileID,1,'int32');            %18
res.yPoints             =   fread(fileID,1,'int32');            %19

res.totalNumberOfPoints =   fread(fileID,1,'int32');            %20

%Distance between points
res.xSpacing            =   fread(fileID,1,'float');            %21
res.ySpacing            =   fread(fileID,1,'float');            %22
res.zSpacing            =   fread(fileID,1,'float');            %23

%Name of axis
res.xName               =   freadChar(fileID,16);               %24
res.yName               =   freadChar(fileID,16);               %25
res.zName               =   freadChar(fileID,16);               %26

%Unit of distance between points
res.xStepUnit           =   freadChar(fileID,16);               %27
res.yStepUnit           =   freadChar(fileID,16);               %28
res.zStepUnit           =   freadChar(fileID,16);               %29

%Unit of axis
res.xLengthUnit         =   freadChar(fileID,16);               %30
res.yLengthUnit         =   freadChar(fileID,16);               %31
res.zLengthUnit         =   freadChar(fileID,16);               %32

%Skaling of distance between points
res.xUnitRatio          =   fread(fileID,1,'float');            %33
res.yUnitRatio          =   fread(fileID,1,'float');            %34
res.zUnitRatio          =   fread(fileID,1,'float');            %35


res.imprint             =   fread(fileID,1,'int16');            %36
res.inverted            =   fread(fileID,1,'int16');            %37
res.levelled            =   fread(fileID,1,'int16');            %38
freadChar(fileID,12);       %obsolete                           %39

%timestamp
res.startSeconds        =   fread(fileID,1,'int16');            %40
res.startMinutes        =   fread(fileID,1,'int16');            %41
res.startHours          =   fread(fileID,1,'int16');            %42
res.startDays           =   fread(fileID,1,'int16');            %43
res.startMonths         =   fread(fileID,1,'int16');            %44
res.startYears          =   fread(fileID,1,'int16');            %45
res.startWeekDay        =   fread(fileID,1,'int16');            %46
res.measurementDuration =   fread(fileID,1,'float');            %47

freadChar(fileID,10);       %obsolete                           %48

%Size of comment field
res.commentSize         =   fread(fileID,1,'int16');            %49
res.privateSize         =   fread(fileID,1,'int16');            %50

res.clientZone          =   freadChar(fileID,128);              %51

%Axis offset
res.xOffset             =   fread(fileID,1,'float');            %52
res.yOffset             =   fread(fileID,1,'float');            %53
res.zOffset             =   fread(fileID,1,'float');            %54

%temperature skale
res.tSpacing            =   fread(fileID,1,'float');            %55
res.tOffset             =   fread(fileID,1,'float');            %56
res.tStepUnit           =   freadChar(fileID,13);               %57
res.tAxisName           =   freadChar(fileID,13);               %58

res.comment             =   freadChar(fileID,res.commentSize);  %59
res.private             =   freadChar(fileID,res.privateSize);  %60

%read datapoints
if res.sizeOfPoints == 16                                       %61
    res.points          =   fread(fileID,'int16');
elseif res.sizeOfPoints == 32
    res.points          =   fread(fileID,'int32');           
end

%reshape datapoints into 2D-Matrix
res.pointsAligned       =   reshape(res.points,[res.xPoints,res.yPoints]);

%Skale datapoints without Offset
res.pointsAligned       =   (res.pointsAligned-res.zMin) * res.zSpacing...
                                / res.zUnitRatio;

%Generate Axis without Offset;
res.xAxis               =   linspace(0,res.xSpacing * res.xPoints ...
                                / res.xUnitRatio, res.xPoints);
res.yAxis               =   linspace(0,res.ySpacing * res.yPoints...
                                / res.yUnitRatio, res.yPoints);

fclose(fileID);
end

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

