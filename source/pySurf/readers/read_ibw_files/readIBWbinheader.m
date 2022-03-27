function H = readIBWbinheader(FFN);
% readIBWbinheader - read BinHeader segment of Igor IBW file
%  H = readIBWbinheader('foo') reads BinHeader of file Foo.ibw
%   IBW version 2,5 only
MAXDIMS = 4;

fid = fopen(FFN,'r');
H.version = fread(fid,1,'int16');
switch H.version,
    case 2,   %    typedef struct BinHeader2 {
        H.wfmSize = fread(fid,1,'uint32'); %  long wfmSize;      // The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
        H.noteSize = fread(fid,1,'uint32'); %  long noteSize;      // The size of the note text.
        H.pictSize = fread(fid,1,'uint32'); %  long pictSize;      // Reserved. Write zero. Ignore on read.
        H.checksum = fread(fid,1,'int16'); %  short checksum;      // Checksum over this header and the wave header.
    case 5,
        H.checksum = fread(fid,1,'short');
        H.wfmSize =  fread(fid,1,'long'); % The size of the WaveHeader5 data structure plus the wave data.
        H.formulaSize =  fread(fid,1,'long');   % The size of the dependency formula, if any.
        H.noteSize =  fread(fid,1,'long');   % The size of the note text.
        H.dataEUnitsSize =  fread(fid,1,'long');  % The size of optional extended data units.
        H.dimEUnitsSize =  fread(fid,MAXDIMS,'long');  % The size of optional extended dimension units.
        H.dimLabelsSize =  fread(fid,MAXDIMS,'long'); % The size of optional dimension labels.
        H.sIndicesSize =  fread(fid,1,'long'); ;     % The size of string indicies if this is a text wave.
        H.optionsSize1 =  fread(fid,1,'long'); ;     % Reserved. Write zero. Ignore on read.
        H.optionsSize2 =  fread(fid,1,'long'); ;     % Reserved. Write zero. Ignore on read.
    otherwise,
end % switch H.Version
fclose(fid);

if ~ismember(H.version,[2 5]),
    error(['Cannot read version ' num2str(H.version) ' IBW files - only versions 2,5 are okay.']);
end
%==FROM IgorBin.h =====================
%  #define MAXDIMS 4
%
% typedef struct BinHeader5 {
%  short version;      // Version number for backwards compatibility.
%  short checksum;      // Checksum over this header and the wave header.
%  long wfmSize;      // The size of the WaveHeader5 data structure plus the wave data.
%  long formulaSize;     // The size of the dependency formula, if any.
%  long noteSize;      // The size of the note text.
%  long dataEUnitsSize;    // The size of optional extended data units.
%  long dimEUnitsSize[MAXDIMS];  // The size of optional extended dimension units.
%  long dimLabelsSize[MAXDIMS];  // The size of optional dimension labels.
%  long sIndicesSize;     // The size of string indicies if this is a text wave.
%  long optionsSize1;     // Reserved. Write zero. Ignore on read.
%  long optionsSize2;     // Reserved. Write zero. Ignore on read.
% } BinHeader5;




