function [BH, H, FFN] = readIBWheaders(FFN);
% readIBWheaders - read BinHeader & WaveHeader segments of Igor IBW file
%  [BH, WH, FFN] = readIBWheaders('foo') reads BinHeader & WaveHeader of
%   file Foo.ibw - Versions 2 and 5 only. FFN is full filename.

% % 1 byte char
% % 2 bytes short
% % 4 bytes int, long, float, Handle, any kind of pointer
% % 8 bytes double
MAXDIMS = 4;
MAX_WAVE_NAME2 = 18; % Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31; % Maximum length of wave name in version 5 files. Does not include the trailing null.
MAX_UNIT_CHARS = 3;


BH = readIBWbinheader(FFN);
fid = fopen(FFN,'r');
switch BH.version
    case 2,
        %         The version 2 file has the following general layout.
        %         BinHeader2 structure: 16 bytes
        %         WaveHeader2 structure excluding wData field; 110 bytes
        %         Wave data: Variable size
        fseek(fid,16,'bof'); % start of wave header of IBW vs 2; % struct WaveHeader2 {
        H.type = fread(fid,1, 'uint16'); %  short type; % See types (e.g. NT_FP64) above. Zero for text waves.
        H.WaveHeader2 = fread(fid,1, 'uint32'); %  struct WaveHeader2 **next;   // Used in memory only. Write zero. Ignore on read.
        H.bname = local_char(fread(fid, MAX_WAVE_NAME2+2, 'char')); %  char bname[MAX_WAVE_NAME2+2];  // Name of wave plus trailing null.
        H.whVersion = fread(fid,1, 'uint16'); %  short whVersion; // Write 0. Ignore on read.
        H.srcFldr = fread(fid,1, 'uint16'); %  short srcFldr;      // Used in memory only. Write zero. Ignore on read.
        H.fileName = fread(fid,1, 'uint32'); %  Handle fileName;     // Used in memory only. Write zero. Ignore on read.
        %
        H.dataUnits = local_char(fread(fid, MAX_UNIT_CHARS+1, 'char')); %  char dataUnits[MAX_UNIT_CHARS+1]; // Natural data units go here - null if none.
        H.xUnits = local_char(fread(fid, MAX_UNIT_CHARS+1, 'char')); %  char xUnits[MAX_UNIT_CHARS+1];  // Natural x-axis units go here - null if none.
        %
        H.npnts = fread(fid,1, 'uint32'); %  long npnts;       // Number of data points in wave.
        %
        H.aModified = fread(fid,1, 'uint16'); %  short aModified;     // Used in memory only. Write zero. Ignore on read.
        H.hsA = fread(fid,1, 'double'); %  double hsA,hsB;      // X value for point p = hsA*p + hsB
        H.hsB = fread(fid,1, 'double'); %  double hsA,hsB;      // X value for point p = hsA*p + hsB
        %
        H.wModified = fread(fid,1, 'uint16'); %  short wModified;     // Used in memory only. Write zero. Ignore on read.
        H.swModified = fread(fid,1, 'uint16'); %  short swModified;     // Used in memory only. Write zero. Ignore on read.
        H.fsValid = fread(fid,1, 'uint16'); %  short fsValid;      // True if full scale values have meaning.
        H.topFullScale = fread(fid,1, 'double'); %  double topFullScale,botFullScale; // The min full scale value for wave.
        H.botFullScale = fread(fid,1, 'double'); %  double topFullScale,botFullScale; // The min full scale value for wave.
        %
        H.useBits = fread(fid,1, 'char'); %  char useBits;      // Used in memory only. Write zero. Ignore on read.
        H.kindBits = fread(fid,1, 'char'); %  char kindBits;      // Reserved. Write zero. Ignore on read.
        H.formula = fread(fid,1, 'uint32'); %  void **formula;      // Used in memory only. Write zero. Ignore on read.
        H.depID = fread(fid,1, 'uint32'); %  long depID;       // Used in memory only. Write zero. Ignore on read.
        H.creationDate = fread(fid,1, '*uint32'); %  unsigned long creationDate;   // DateTime of creation. Not used in version 1 files.
        H.platform = fread(fid,1, 'uint8'); %  unsigned char platform;    // 0=unspecified, 1=Macintosh, 2=Windows; Added for Igor Pro 5.5.
%        dummy = fread(fid,1, 'uint8'); %  align
        H.wUnused = fread(fid,1, 'uint8'); %  char wUnused[1];     // Reserved. Write zero. Ignore on read.
        %
        H.modDate = fread(fid,1, '*uint32'); %  unsigned long modDate;    // DateTime of last modification.
        H.waveNoteH = fread(fid,1, 'uint32'); %  Handle waveNoteH;     // Used in memory only. Write zero. Ignore on read.
        %
        H.wData = fread(fid,4, 'single').'; %  float wData[4];      // The start of the array of waveform data.
    case 5,
        %         The version 5 file has the following general layout.
        %         BinHeader5 structure: 64 bytes
        %         WaveHeader5 structure excluding wData field: 320 bytes
        %         Wave data: Variable size
        fseek(fid,64,'bof'); % start of wave header of IBW vs 5
        %
        H.next = fread(fid,1, 'int32');  % link to next wave in linked list.
        %
        H.creationDate = fread(fid,1, 'uint32');  % DateTime of creation.
        H.modDate = fread(fid,1, 'uint32');  % DateTime of creation.
        %
        H.npnts = fread(fid,1, 'int32');  % Total number of points (multiply dimensions up to first zero).
        H.type = fread(fid,1, 'uint16'); % See types (e.g. NT_FP64) above. Zero for text waves.
        H.dLock = fread(fid, 1, 'int16'); % Reserved. Write zero. Ignore on read.
        %
        H.whpad1 = fread(fid, 6, 'char').'; % Reserved. Write zero. Ignore on read.
        H.whVersion = fread(fid, 1, 'int16'); % Write 1. Ignore on read.
        H.bname = local_char(fread(fid, MAX_WAVE_NAME5+1, 'char')); % Name of wave plus trailing null.
        H.whpad2 = fread(fid, 1, 'int32'); % long Reserved. Write zero. Ignore on read.
        H.dFolder = fread(fid, 1, 'int32'); % ptr Used in memory only. Write zero. Ignore on read.

        %  % Dimensioning info. [0] == rows, [1] == cols etc
        H.nDim = fread(fid, MAXDIMS, 'int32').'; % long nDim[MAXDIMS] Number of of items in a dimension -- 0 means no data.
        H.sfA = fread(fid, MAXDIMS, 'double').'; % double sfA[MAXDIMS]; Index value for element e of dimension d = sfA[d]*e + sfB[d].
        H.sfB = fread(fid, MAXDIMS, 'double').'; % double sfB[MAXDIMS];%  double sfB[MAXDIMS];
        % %
        % %  % SI units
        H.dataUnits = local_char(fread(fid, MAX_UNIT_CHARS+1, 'char')); %  char dataUnits[MAX_UNIT_CHARS+1];   % Natural data units go here - null if none.
        H.dimUnits = fread(fid, MAXDIMS*(MAX_UNIT_CHARS+1), '*char').'; % %  char dimUnits[MAXDIMS][MAX_UNIT_CHARS+1]; % Natural dimension units go here - null if none.
        % %
        H.fsValid = fread(fid, 1, 'uint16'); % %  short fsValid;      % TRUE if full scale values have meaning.
        H.whpad3 = fread(fid, 1, 'int16'); % %  short whpad3;      % Reserved. Write zero. Ignore on read.
        H.topFullScale = fread(fid, 1, 'double');
        H.botFullScale = fread(fid, 1, 'double'); % %  double topFullScale,botFullScale; % The max and max full scale value for wave.
        % %
        H.dataEUnits = fread(fid, 1, 'int32'); % %  Handle dataEUnits;     % Used in memory only. Write zero. Ignore on read.
        H.dimEUnits = fread(fid, MAXDIMS, 'int32').'; % %  Handle dimEUnits[MAXDIMS];   % Used in memory only. Write zero. Ignore on read.
        H.dimLabels = fread(fid, MAXDIMS, 'int32').'; % %  Handle dimLabels[MAXDIMS];   % Used in memory only. Write zero. Ignore on read.
        % %
        H.waveNoteH = fread(fid, 1, 'int32'); % %  Handle waveNoteH;     % Used in memory only. Write zero. Ignore on read.
        % %
        H.platform = fread(fid, 1, 'uint8'); % %  unsigned char platform;    % 0=unspecified, 1=Macintosh, 2=Windows; Added for Igor Pro 5.
        H.skip_junk________ = fread(fid,80,'char').';
        % H. = fread(fid, 1, '');
        % %  unsigned char spare[3];
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  long whUnused[13];     % Reserved. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  long vRefNum, dirID;    % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  % The following stuff is considered private to Igor.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  short aModified;     % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  short wModified;     % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  short swModified;     % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  char useBits;      % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  char kindBits;      % Reserved. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  void **formula;      % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  long depID;       % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  short whpad4;      % Reserved. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  short srcFldr;      % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %  Handle fileName;     % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        % H. = fread(fid, 1, '');
        % %  long **sIndices;     % Used in memory only. Write zero. Ignore on read.
        % H. = fread(fid, 1, '');
        % %
        H.wData = fread(fid, 1, 'single');
        % %  float wData[1];      % The start of the array of data. Must be 64 bit aligned.
        % H. = fread(fid, 1, '');
        % };
end % switch BH.version
H.EOH_pos = ftell(fid);




fclose(fid);

%==FROM IgorBin.h =====================
%  #define MAXDIMS 4

% % #define MAX_WAVE_NAME5 31 // Maximum length of wave name in version 5 files. Does not include the trailing null.
% % #define MAX_UNIT_CHARS 3

% % #define NT_CMPLX 1   % Complex numbers.
% % #define NT_FP32 2   % 32 bit fp numbers.
% % #define NT_FP64 4   % 64 bit fp numbers.
% % #define NT_I8 8    % 8 bit signed integer. Requires Igor Pro 2.0 or later.
% % #define NT_I16  0x10  % 16 bit integer numbers. Requires Igor Pro 2.0 or later.
% % #define NT_I32  0x20  % 32 bit integer numbers. Requires Igor Pro 2.0 or later.
% % #define NT_UNSIGNED 0x40 % Makes above signed integers unsigned. Requires Igor Pro 3.0 or later.
%
% struct WaveHeader5 {
%  struct WaveHeader5 **next;   % link to next wave in linked list.
%
%  unsigned long creationDate;   % DateTime of creation.
%  unsigned long modDate;    % DateTime of last modification.
%
%  long npnts;       % Total number of points (multiply dimensions up to first zero).
%  short type;       % See types (e.g. NT_FP64) above. Zero for text waves.
%  short dLock;      % Reserved. Write zero. Ignore on read.
%
%  char whpad1[6];      % Reserved. Write zero. Ignore on read.
%  short whVersion;     % Write 1. Ignore on read.
%  char bname[MAX_WAVE_NAME5+1];  % Name of wave plus trailing null.
%  long whpad2;      % Reserved. Write zero. Ignore on read.
%  struct DataFolder **dFolder;  % Used in memory only. Write zero. Ignore on read.
%
%  % Dimensioning info. [0] == rows, [1] == cols etc
%  long nDim[MAXDIMS];     % Number of of items in a dimension -- 0 means no data.
%  double sfA[MAXDIMS];    % Index value for element e of dimension d = sfA[d]*e + sfB[d].
%  double sfB[MAXDIMS];
%
%  % SI units
%  char dataUnits[MAX_UNIT_CHARS+1];   % Natural data units go here - null if none.
%  char dimUnits[MAXDIMS][MAX_UNIT_CHARS+1]; % Natural dimension units go here - null if none.
%
%  short fsValid;      % TRUE if full scale values have meaning.
%  short whpad3;      % Reserved. Write zero. Ignore on read.
%  double topFullScale,botFullScale; % The max and max full scale value for wave.
%
%  Handle dataEUnits;     % Used in memory only. Write zero. Ignore on read.
%  Handle dimEUnits[MAXDIMS];   % Used in memory only. Write zero. Ignore on read.
%  Handle dimLabels[MAXDIMS];   % Used in memory only. Write zero. Ignore on read.
%
%  Handle waveNoteH;     % Used in memory only. Write zero. Ignore on read.
%
%  unsigned char platform;    % 0=unspecified, 1=Macintosh, 2=Windows; Added for Igor Pro 5.5.
%  unsigned char spare[3];
%
%  long whUnused[13];     % Reserved. Write zero. Ignore on read.
%
%  long vRefNum, dirID;    % Used in memory only. Write zero. Ignore on read.
%
%  % The following stuff is considered private to Igor.
%
%  short aModified;     % Used in memory only. Write zero. Ignore on read.
%  short wModified;     % Used in memory only. Write zero. Ignore on read.
%  short swModified;     % Used in memory only. Write zero. Ignore on read.
%
%  char useBits;      % Used in memory only. Write zero. Ignore on read.
%  char kindBits;      % Reserved. Write zero. Ignore on read.
%  void **formula;      % Used in memory only. Write zero. Ignore on read.
%  long depID;       % Used in memory only. Write zero. Ignore on read.
%
%  short whpad4;      % Reserved. Write zero. Ignore on read.
%  short srcFldr;      % Used in memory only. Write zero. Ignore on read.
%  Handle fileName;     % Used in memory only. Write zero. Ignore on read.
%
%  long **sIndices;     % Used in memory only. Write zero. Ignore on read.
%
%  float wData[1];      % The start of the array of data. Must be 64 bit aligned.
% };

function s=local_char(S);
% null-terminated string -> char string
S = S(:).';
S = S(S~=0);
s=char(S);








