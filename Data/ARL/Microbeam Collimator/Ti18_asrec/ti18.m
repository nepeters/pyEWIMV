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
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% path to files
pname = '/home/nate/wimv/ARL/Fulin/Ti-64';

% which files to be imported
fname = {...
  [pname '/beta_110_2theta=39.355.xrdml'],...
  [pname '/beta_200_56.790.xrdml'],...
  [pname '/beta_220_85.413.xrdml'],...
  };

% background
% pname = '/home/nate/wimv/ARL/Microbeam Collimator/Ti18_asrec';
fname_bg = {...
  [pname '/32_ bkgd.xrdml'],...
  [pname '/47_bkgd.xrdml'],...
  [pname '/67_bkgd.xrdml'],...
  };

% defocusing
pname = '/home/nate/wimv/ARL/Fulin';
fname_def = {...
  [pname '/defoc_tistandard_35.xrdml'],...
  [pname '/defoc_tistandard_35.xrdml'],...
  [pname '/defoc_tistandard_35.xrdml'],...
  };

%% Specify Miller Indice

h = { ...
  Miller(1,1,0,CS),...
  Miller(2,0,0,CS),...
  Miller(2,1,1,CS),...
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

plotPDF(odf,h,'contourf',0:1:5)
