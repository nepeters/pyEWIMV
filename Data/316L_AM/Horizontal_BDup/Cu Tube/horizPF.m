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
pname = 'D:\UVA\Research\AM_316L\XRD\Horizontal_BDup';

% which files to be imported
fname = {...
  [pname '\110_pf_AM316Lhor.xrdml'],...
  [pname '\200_pf_AM316Lhor.xrdml'],...
  [pname '\211_pf_AM316Lhor.xrdml'],...
  [pname '\211_pf_AM316Lhor_redo.xrdml'],...
  };

% background
fname_bg = {...
  [pname '\Bkgd_47_AM316Lhor.xrdml'],...
  [pname '\Bkgd_47_AM316Lhor.xrdml'],...
  [pname '\Bkgd_70_AM316Lhor.xrdml'],...
  [pname '\Bkgd_70_AM316Lhor.xrdml'],...
  };

%% Specify Miller Indice

h = { ...
  Miller(1,1,0,CS),...
  Miller(2,0,0,CS),...
  Miller(2,1,1,CS),...
  Miller(2,1,1,CS),...
  };

%% Import the Data

hP = { ...
  Miller(1,1,0,CS),...
  Miller(2,0,0,CS),...
  Miller(2,1,1,CS),...
  };

% create a Pole Figure variable containing the data
pf = loadPoleFigure(fname,h,CS,SS,'interface','xrdml');

% background
pf_bg = loadPoleFigure(fname_bg,h,CS,SS,'interface','xrdml');

% correct data
pf = correct(pf,'bg',pf_bg);

odf = calcODF(pf{1:2},'silent');

figure
plotPDF(odf,h(1:2),'contourf');

% plot
figure
plot(pf,'contourf')

