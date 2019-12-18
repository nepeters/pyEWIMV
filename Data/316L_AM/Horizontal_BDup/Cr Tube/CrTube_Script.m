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
pname = 'D:\UVA\Research\AM_316L\XRD\Horizontal_BDup\Cr Tube';

% which files to be imported
fname = {...
  [pname '\110_pf_316L_horiz.xrdml'],...
  [pname '\200_pf_316L_horiz.xrdml'],...
  [pname '\211_pf_316L_horiz.xrdml'],...
  };

% background
pname = 'D:\UVA\Research\AM_316L\XRD\Horizontal_BDup\Cr Tube';
fname_bg = {...
  [pname '\Bkgd_73_316L_horiz.xrdml'],...
  [pname '\Bkgd_100_316L_horiz.xrdml'],...
  [pname '\Bkgd_100_316L_horiz2.xrdml'],...
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

% correct data
pf = correct(pf,'bg',pf_bg);

odf = calcODF(pf,'silent');


figure
plotPDF(odf,h,'contourf');
CLim(gcm,[0 3.5]);
figure,
plotIPDF(odf,xvector)
CLim(gcm,[0 3.5]);
figure,
plotIPDF(odf,yvector)
CLim(gcm,[0 3.5]);
figure,
plotIPDF(odf,zvector)
CLim(gcm,[0 3.5]);
