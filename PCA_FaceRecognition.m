

Image_path = uigetdir('C:\Users\ \Documents\MATLAB'); 
%dir name lists the files and folders that match the string name
list_image = dir(Image_path);
img_count = 0;
%Counting the number of Images in the folder specified by
%Image_path(Training Image)
arrname = zeros(10); % array of 10 X 10
for i =1 : size(list_image,1)
    if not(strcmp(list_image(i).name,'.')|strcmp(list_image(i).name,'..'))
       % arrname{i} = int2str(list_image(i).name);
        img_count = img_count + 1;
    end
end
img_count;

X=[];
for i = 1: img_count
    each_img_name = strcat(Image_path,'\',int2str(i),'.jpg');
    img = imread(each_img_name);
    gray_img = rgb2gray(img); %converting RGB in to grayscale image
   % imshow(rgb_img);
    [row,co] = size(gray_img);%Reshape read by column thats why transpose is neccesary
    temp = reshape(gray_img',row*co,1); % Here change the row into column 
    X = horzcat(X,temp); %Creating the mn x p matrix by adding mn x 1 column vetor
end

%M = mean(A,dim) returns the mean values for elements along the dimension of A specified by scalar dim.
%For matrices, mean(A,2) is a column vector containing the mean value of each row.
mean_image = mean(X,2); % mn x 1 matrix ( Mean of all image sum(Xi)/P i = 1:P 
%mean_face = reshape(mean_image,100,[]);
image_cnt = size(X,2);

%Subtract all Image_vectors(X) from the mean Image vector(Mean_image) and
%store it into A i.e A = (Xi - mean_image); 
A = [];
%A1 = X1 - mean_image , A2 = X2 - mean_image, A3 = X3 - mean_image ;
for i = 1:image_cnt
    temp = (double(X(:,i)) - mean_image); % X(:,i) access the elements of ith column
    A = horzcat(A ,temp);
end


Co_var = A'*A;
[eig_vec,eig_val] = eig(Co_var); %EigenVectors & EigenValues

 
PComp = []; %% Principal Components i.e Feature Vectors (Nothing but Only K 
%Eigenvectors.
for i= 1:size(eig_vec,2)
    if( eig_val(i,i) > 1 )
        PComp = [PComp,eig_vec(:,i)];
    end
end

%% Computing the eigen Face by Projecting the 
%each mean aligned Face i.e A (mn * p) to the generated Feature Vector 
%%i.e PComp (p*k) eigenface will have dimension k*mn or we can think of it
%%other ways as well ;) .
eigen_faces = A * PComp; % Step 7
for i= 1:20
B = reshape(eigen_faces(:,i),[100,100]);
new_frame = B;
filename = (['Eigen_',num2str(i),'.jpg']);
imwrite(new_frame, filename);
end

%Step 8
%Generate Signature of each Face - 
%For generating signature face ,project each mean aligned image (mn * p ) to the 
%eigen Faces (mn * k) .
project_trainimg = [];
for i =1:size(eigen_faces,2)
    temp = eigen_faces' * A(:,i); %% k*mn = mn * p i.e signat is k*p
    project_trainimg = horzcat(project_trainimg,temp);
end

%%NOW COMES THE TESTING PART 
%%Firstly We will take an testing image as an input , Make it a column
%%Vector say I1(mnx1), Do the mean zero by subtracting the mean face(mnX1) to
%%this Test Face ,Say I2(mnx1).
%%Project This Mean Aligned Face To EigenFaces, we will get the projected
%%Test Face i.e I3 = eigen_face'(kXmn) * I2(mnX1)
%%Now We have Projected Test Face I3 and Signature of each face signat,
%%Will calculate the Euclidean Distance between them. 

Test_Image_Path = uigetdir('C:\Users\Jitendra Mohan\Documents\MATLAB');
Timage_nam = strcat(Test_Image_Path,'\','4.jpg');
tem = imread(Timage_nam);
testing_image = rgb2gray(tem);
[r,c] = size(testing_image);
testing_image = reshape(testing_image',r*c,1); %%Making it a Column Vector (mnX1)
%imshow(testing_image);
I2 = mean_image - double(testing_image);%% Step 2
proj_testimg = eigen_faces' * I2; %% Projected Test Face Step 3

%%calculating & comparing the euclidian distance FinalStep
euclidean_dist = [];
for i=1:size(eigen_faces,2)
    temp = (norm(proj_testimg-project_trainimg(:,i)))^2;
    euclidean_dist = horzcat(euclidean_dist,temp);
end

[euclide_dist_min index] = min(euclidean_dist);
recognized_image = strcat(int2str(index));
s_img = strcat('C:\Users\Jitendra Mohan\Documents\MATLAB\train','\',recognized_image,'.jpg');
sel_img = imread(s_img);
sel_img = rgb2gray(sel_img);
imshow(sel_img);
title('Recognized Image');
test_img = imread(Timage_nam);
test_img = rgb2gray(test_img);
figure,imshow(test_img);
title('Testing Image');
