%% Import Script for PoleFigure Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = crystalSymmetry('m-3m', [1 1 1], 'color', 'light blue');

% specimen symmetry
SS = specimenSymmetry('1');

% plotting convention
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% path to files
pname = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/wimv2/Data';

% which files to be imported
fname = {...
  [pname '/111pf_2T=38.xrdml'],...
  [pname '/200pf_2T=45.xrdml'],...
  [pname '/220pf_2theta=65.xrdml'],...
  };

% background
pname = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/wimv2/Data';
fname_bg = {...
  [pname '/32.5_ bkgd.xrdml'],...
  [pname '/32.5_ bkgd.xrdml'],...
  [pname '/55_bkgd.xrdml'],...
  };

% defocusing
pname = '/media/nate/2E7481AA7481757D/Users/Nate/Dropbox/wimv2/Data';
fname_def = {...
  [pname '/defocus_38.xrdml'],...
  [pname '/defocus_45.xrdml'],...
  [pname '/defocus_65.xrdml'],...
  };

%% Specify Miller Indice

h = { ...
  Miller(1,1,1,CS),...
  Miller(2,0,0,CS),...
  Miller(2,2,0,CS),...
  };

%% Import the Data

% create a Pole Figure variable containing the data
pf = loadPoleFigure(fname,h,CS,SS,'interface','xrdml');

% background
pf_bg = loadPoleFigure(fname_bg,h,CS,SS,'interface','xrdml');

% defocussing
pf_def = loadPoleFigure(fname_def,h,CS,SS,'interface','xrdml');

% correct data
pf = correct(pf,'bg',pf_bg,'def',pf_def);

odf = calcODF(pf);

figure
plotPDF(odf,h,'contourf',0:1:12)