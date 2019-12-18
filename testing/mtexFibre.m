
cs = crystalSymmetry('m-3m');
h = Miller(1,1,1,cs);
r = vector3d(0.5,0.24,0.343,cs);

f = fibre(h,r);

nPoints = 1000;

omega = linspace(0,2*pi,nPoints);

rot = rotation.id(nPoints,length(f.h));

o1 = quaternion(f.o1);
o2 = quaternion(f.o2);

for i = 1:length(f.h)
    
    rot(:,i) = rotation('axis',r(i),'angle',omega) .* o1(i);
    
end

ori = orientation(rot,cs);
