close all, clear all
A = dlmread('test.data');
meanA = repmat(sum(A,1),length(A),1)/length(A);
B = A - meanA;  % only use ground truth mean for test, may further try A1's mean...
mse_F = compute_F(A,B); % missing the frames: 71-75
mse_F1 = compute_F1(A,B);
mse_T = compute_T(A,B); % missing the joint: 70-72
mse_T1 = compute_T1(A,B);

[mse_T0,rate_T0] = compute_T0(A,B); % missing entries randomly instead of missing whole columns/rows
[mse_F0,rate_F0] = compute_F0(A,B);

function [mse,rate] = compute_T0(A,B)
B = B';
M = B*B';
[u,d,v]=svd(M);
[n,m] = size(B(:,1:100));
s = 0;
while s<15
    mask = rand(n,m);
    mask(find(mask>0.75))=1;
    mask(find(mask<1))=0;
    s= min([min(sum(mask,1)),min(sum(mask,2))]);
end
rate = sum(sum(mask))/(n*m);
Bs = [];
B0 = [];
u0 = [];
for i=1:4
    tmp = B(:,(i-1)*100+1:i*100);
    Bs = [Bs,tmp];
    tmp = tmp.*mask;
    B0 = [B0,tmp];
    [ut,dt,vt] = svd(tmp*tmp');
    u0 = [u0,ut];
end
        
alpha = [];
alpha0 = [];
for i=1:4
    alpha =  [alpha, u'                  *Bs(:,(i-1)*100+1:i*100)];
    alpha0 = [alpha0,u0(:,(i-1)*n+1:i*n)'*B0(:,(i-1)*100+1:i*100)];
end

T = (alpha*alpha0')*inv(alpha0*alpha0');

mean1 = repmat(sum(A(402:end,:),1),100,1)/100;
%mean0 = repmat(sum(A([400:469,75:end],:),1),97 ,1)/97;
A1 = (A(402:end,:) - mean1)';
GT = A(402:end,:)';
A10 = A1.*mask;

[u10,d10,v10] = svd(A10*A10');

alpha1 = u10'*A10;

A11 = u*T*alpha1;
Astar = A11 + mean1';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end


function mse = compute_T1(A,B)
B = B';
M = B*B';
[u,d,v]=svd(M);

mean1 = repmat(sum(A(401:end,:),1),101,1)/101;
% mean1 = 0;
% for i=1:25
%     tmp = A(401:end,(i-1)*3+1:i*3);
%     mean1 = mean1 + tmp;
% end
% mean1 = mean1/i;
% mean1 = repmat(mean1,1,i);

A1 = (A(401:end,:) - mean1)';
GT = A(401:end,:)';
GT = GT(70:72,:);
A10 = A1([1:69,73:end],:);

alpha = u([1:69,73:end],:)'*inv(u([1:69,73:end],:)*u([1:69,73:end],:)')*A10;

A11 = u(70:72,:)*alpha;
Astar = A11 + mean1(:,70:72)';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end


function mse = compute_T(A,B)
B = B';
M = B*B';
[u,d,v]=svd(M);

Bs = [];
B0 = [];
u0 = [];
for i=1:4
    tmp = B(:,(i-1)*100+1:i*100);
    Bs = [Bs,tmp];
    tmp = tmp([1:69,73:end],:);
    B0 = [B0,tmp];
    [ut,dt,vt] = svd(tmp*tmp');
    u0 = [u0,ut];
end
        
alpha = [];
alpha0 = [];
for i=1:4
    alpha =  [alpha, u'                    *Bs(:,(i-1)*100+1:i*100)];
    alpha0 = [alpha0,u0(:,(i-1)*72+1:i*72)'*B0(:,(i-1)*100+1:i*100)];
end

T = (alpha*alpha0')*inv(alpha0*alpha0');

mean1 = repmat(sum(A(402:end,:),1),100,1)/100;
%mean0 = repmat(sum(A([400:469,75:end],:),1),97 ,1)/97;
A1 = (A(402:end,:) - mean1)';
GT = A(402:end,:)';
GT = GT(70:72,:);
A10 = A1([1:69,73:end],:);

[u10,d10,v10] = svd(A10*A10');

alpha1 = u10'*A10;

A11 = u(70:72,:)*T*alpha1;
Astar = A11 + mean1(:,70:72)';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end


function mse = compute_F(A,B)
B = [B(1:102,:),B(100:201,:),B(200:301,:),B(300:401,:)]';

M = B'*B;
[u,d,v]=svd(M);
u = u(:,1:75);

B0 = [];
u0 = [];
for i=1:4
    tmp = B((i-1)*75+1:i*75,:);
    tmp = tmp(:,[1:70,76:end]);
    B0 = [B0;tmp];
    [ut,dt,vt] = svd(tmp'*tmp);
    u0 = [u0;ut(:,1:75)];
end
        
alpha = [];
alpha0 = [];
for i=1:4
    alpha = [alpha;B((i-1)*75+1:i*75,:)*u];
    alpha0 = [alpha0;B0((i-1)*75+1:i*75,:)*u0((i-1)*97+1:i*97,:)];
end

F = inv(alpha0'*alpha0)*(alpha0'*alpha);

mean1 = repmat(sum(A(400:end          ,:),1),102,1)/102;
mean0 = repmat(sum(A([400:469,475:end],:),1),97 ,1)/97;
A1 = (A(400:end,:) - mean1)';
GT = A(400:end,:)';
GT = GT(:,71:75);
A10 = A1(:,[1:70,76:end]);

[u10,d10,v10] = svd(A10'*A10);
u10 = u10(:,1:75);
alpha1 = A10*u10;

A11 = alpha1*F*u(71:75,:)';
Astar = A11 + mean1(71:75,:)';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end

function [mse,rate] = compute_F0(A,B)
B = [B(1:102,:),B(100:201,:),B(200:301,:),B(300:401,:)]';

M = B'*B;
[u,d,v]=svd(M);
u = u(:,1:75);

[m,n]=size(A(1:102,:));
s = 0;
while s<15
    mask = rand(n,m);
    mask(find(mask>0.75))=1;
    mask(find(mask<1))=0;
    s= min([min(sum(mask,1)),min(sum(mask,2))]);
end
rate = sum(sum(mask))/(n*m);

B0 = [];
u0 = [];
for i=1:4
    tmp = B((i-1)*75+1:i*75,:);
    tmp = tmp.*mask;
    B0 = [B0;tmp];
    [ut,dt,vt] = svd(tmp'*tmp);
    u0 = [u0;ut(:,1:75)];
end
        
alpha = [];
alpha0 = [];
for i=1:4
    alpha = [alpha;B((i-1)*75+1:i*75,:)*u];
    alpha0 = [alpha0;B0((i-1)*75+1:i*75,:)*u0((i-1)*m+1:i*m,:)];
end

F = inv(alpha0'*alpha0)*(alpha0'*alpha);

mean1 = repmat(sum(A(400:end          ,:),1),102,1)/102;

A1 = (A(400:end,:) - mean1)';
GT = A(400:end,:)';

A10 = A1.*mask;

[u10,d10,v10] = svd(A10'*A10);
u10 = u10(:,1:75);
alpha1 = A10*u10;

A11 = alpha1*F*u';
Astar = A11 + mean1';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end

function mse = compute_F1(A,B)
B = [B(1:102,:),B(100:201,:),B(200:301,:),B(300:401,:)]';

M = B'*B;
[u,d,v]=svd(M);
u = u(:,1:75);

mean1 = repmat(sum(A(400:end          ,:),1),102,1)/102;
%mean0 = repmat(sum(A([400:469,475:end],:),1),97 ,1)/97;
A1 = (A(400:end,:) - mean1)';
GT = A(400:end,:)';
GT = GT(:,71:75);
A10 = A1(:,[1:70,76:end]);

alpha = A10*u([1:70,76:end],:)*inv(u([1:70,76:end],:)'*u([1:70,76:end],:));

A11 = alpha*u(71:75,:)';
Astar = A11 + mean1(71:75,:)';
[n,m] = size(Astar);
mse = sqrt(sum(sum((Astar-GT).*(Astar-GT))))/(n*m);
end