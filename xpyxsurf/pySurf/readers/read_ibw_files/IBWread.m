function D = IBWread(FN);
% IBWread - read Igor wave from IBW file
%   D = IBWread('Foo.ibw') reads Igor file Foo.ibw into struct D.

if exist('fullfilename', 'file'), FFN = fullfilename(FN, cd, '.ibw');
else, FFN = FN;
end

% read headers
[D.binHeader, D.waveHeader, FFN] = readIBWheaders(FN);
fid = fopen(FFN,'r');
fid = fopen(FFN,'r');
% 	For numeric waves, the type field is interpreted bitwise. One of the following bits, as represented by symbols defined in IgorBin.h, will be set:
% 
% % #define NT_CMPLX 1   % Complex numbers.
% % #define NT_FP32 2   % 32 bit fp numbers.
% % #define NT_FP64 4   % 64 bit fp numbers.
% % #define NT_I8 8    % 8 bit signed integer. Requires Igor Pro 2.0 or later.
% % #define NT_I16  0x10  % 16 bit integer numbers. Requires Igor Pro 2.0 or later.
% % #define NT_I32  0x20  % 32 bit integer numbers. Requires Igor Pro 2.0 or later.
% % #define NT_UNSIGNED 0x40 % Makes above signed integers unsigned. Requires Igor Pro 3.0 or later.
% 	NT_FP64		64-bit floating point (double)
% 	NT_FP32		32-bit floating point (float)
% 	NT_I32		32 bit integer (long)
% 	NT_16			16 bit integer (short)
% 	NT_I8			8 bit integer (char)
datatype = D.waveHeader.type;
CMPLX = rem(datatype,2);
if CMPLX, datatype = datatype-1; end
switch datatype,
    case 0,     prec = '*char';
    case 2,     prec = 'single';
    case 4, prec = 'double';
    case 8, prec = 'int8';
    case 16, prec = 'int16';
    case 32, prec = 'int32';
    case 64+8, prec = 'uint8';
    case 64+16, prec = 'uint16';
    case 64+32, prec = 'uint32';
    otherwise, error('Invalid numerical datatype.');
end

D.bname = D.waveHeader.bname;
% The date/time is store as seconds since midnight, January 1, 1904.
D.creationDate = local_igorTime2vec(D.waveHeader.creationDate);
D.modDate = local_igorTime2vec(D.waveHeader.creationDate);
switch D.binHeader.version,
    case 2,
        %         The version 2 file has the following general layout.
        %         BinHeader2 structure: 16 bytes
        %         WaveHeader2 structure excluding wData field; 110 bytes
        %         Wave data: Variable size
        D.Nsam = D.waveHeader.npnts;
        D.Ndim = 1;
        % X value for point p = hsA*p + hsB
        D.dx = D.waveHeader.hsA;
        D.x0 = D.waveHeader.hsB;
        D.x1 = D.x0+D.Nsam*D.dx;
        fseek(fid, 16+110, 'bof');
        NN = D.Nsam*(1+double(CMPLX)); % if complex, read twice as many numbers
        D.y = fread(fid, NN, prec);
        if CMPLX, D.y = D.y(1:2:end) + i*D.y(2:2:end); end; % Re & Im values are interleaved
    case 5,
        % The version 5 file has the following general layout.
        % BinHeader5 structure 64 bytes
        % WaveHeader5 structure excluding wData field 320 bytes
        % Wave data Variable size
        % Optional wave dependency formula Variable size
        % Optional wave note data Variable size
        D.Nsam = D.waveHeader.npnts;
        D.Ndim = sum(D.waveHeader.nDim>0);
        % X offset & spacing (documentation is obscure)
        D.dx = D.waveHeader.sfA(1:D.Ndim);
        D.x0 = D.waveHeader.sfB(1:D.Ndim);
        D.x1 = nan; % D.x0+D.Nsam*D.dx;
        fseek(fid, 64+320, 'bof');
        NN = D.Nsam*(1+double(CMPLX)); % if complex, read twice as many numbers
        if isequal(0,datatype), % text data
            % Number of bytes of wave data = wfmSize - (sizeof(WaveHeader5) - 4)
            NN = D.binHeader.wfmSize-320;
            D.y = fread(fid, NN, prec)';
        else, % numerical data
            D.y = fread(fid, NN, prec);
        end
        if CMPLX, D.y = D.y(1:2:end) + i*D.y(2:2:end); end; % Re & Im values are interleaved
        if ~isempty(D.y) && D.Ndim>1,
            D.y = reshape(D.y, D.waveHeader.nDim(1:D.Ndim));
        end
end
D.WaveNotes = local_readNotes(fid, D.binHeader);
if isequal(0,datatype), % partition text data into lines using string indices stored @ eof 
    D = local_partition_text(fid, D);
end
fseek(fid,0,'eof');
D.fileSize = ftell(fid);
fclose(fid);


%===============================
function DV=local_igorTime2vec(it);
% igor time -> date vector a la Matlab
% The date/time is store as seconds since midnight, January 1, 1904.
Nsec_4year = 126230400;
idate_1988 = 21*uint32(Nsec_4year); % # seconds between 1-jan-1904 & 1-jan-1988
it = double(it-idate_1988); % smaller numbers-> use ordinary numbers ("doubles")
M4 = floor(it/Nsec_4year);
YR = double(1988+4*M4);
it = double(it-M4*Nsec_4year);
last_it = it;
while 1, % subtract years as long as they fit
    s = etime([YR+1 1 1 0 0 0], [YR 1 1 0 0 0]);
    if it<s, break; end % we went too far
    YR = YR+1;
    it = it-s;
end
MNTH = 1;
while 1, % subtract months as long as they fit
    s = etime([YR MNTH+1 1 0 0 0], [YR MNTH 1 0 0 0]);
    if it<s, break; end % we went too far
    MNTH = MNTH+1;
    it = it-s;
end
DAY = 1;
while 1, % subtract days as long as they fit
    s = etime([YR MNTH DAY+1 0 0 0], [YR MNTH DAY 0 0 0]);
    if it<s, break; end % we went too far
    DAY = DAY+1;
    it = it-s;
end
HR = floor(it/60^2); it = it-HR*60^2;
MIN = floor(it/60); SEC = it-MIN*60;
DV = [YR, MNTH, DAY, HR, MIN, SEC];
if ~isequal(last_it, etime(DV, [1988+4*M4 1 1 0 0 0])),
    error('date error');
end

function Nts = local_readNotes(fid, B);
switch B.version, 
    case 2,
        % BinHeader2 structure 16 bytes
        % WaveHeader2 structure excluding wData field 110 bytes
        % Wave data Variable size
        % 16 bytes of padding 16 bytes
        dum = fread(fid,16,'char');
        % Optional wave note data
        Nts = fread(fid, B.noteSize, '*char').';
    case 5,
        % BinHeader5 structure 64 bytes
        % WaveHeader5 structure excluding wData field 320 bytes
        % Wave data Variable size
        % Optional wave dependency formula Variable size
        dum = fread(fid,B.formulaSize, 'char');
        % Optional wave note data Variable size
        Nts = fread(fid, B.noteSize, '*char').';
end

function D = local_partition_text(fid, D);
% read sizeindices and chop text accordingly, must be called after reading
% notes
bH = D.binHeader;
% skip other stuff in file
% The version 5 file has the following general layout.
% BinHeader5 structure	64 bytes
% WaveHeader5 structure excluding wData field	320 bytes
% Wave data	Variable size
% Optional wave dependency formula	Variable size
% Optional wave note data	Variable size
% Optional extended data units data	Variable size
% Optional extended dimension units data	Variable size
% Optional dimension label data	Variable size
% String indices used for text waves only	Variable size
dum=fseek(fid, -bH.sIndicesSize, 'eof'); % string indices are last in file; it's easier to refer from end-of-file
% bH.sIndicesSize; # bytes of string indices stored for the wave if the wave type is text. 
% String indices are used to determine the offset in the file and the number of bytes for each element of the text wave.
sIdx = [0 fread(fid, bH.sIndicesSize/4, 'uint32').']; 
for ii=1:numel(sIdx)-1,
    STR{ii,1} = D.y(sIdx(ii)+1:sIdx(ii+1));
end
D.y = STR;


% long sIndicesSize; The size of string indicies if this is a text wave.






