function main
	addpath('libsvm-3.18\matlab');
	data=load('face_data.mat');
	for i=1:size(data.image,2)
		X(:,i) = data.image{i}(:);
	end

	X = double(X);
	eigenvecs = pca_fun(X', 20);
	for i=1:5
		 figure;
		 imshow(uint8(reshape((eigenvecs(:,i)-min(eigenvecs(:,i)))*255/(max(eigenvecs(:,i))-min(eigenvecs(:,i))),[50,50])));
	end

	C = [4^-7,4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,4^0,4^1,4^2,4^3,4^4,4^5,4^6,4^7];
	D = [20,50,100,200];
	personID = (data.personID)';
	subset = data.subsetID;

	for z=1:4
		total_accuracy = 0;
		avg_accuracy = 0;
		max_accuracy = 0;
		total_test_accuracy = 0;
		
		eigenvecs = pca_fun(X', D(z));
		proj_image = X'*eigenvecs;
		
		for i=1:5
			train_data = proj_image(subset ~= i,:);
			train_label = personID(subset ~= i,:);
			test_data = proj_image(subset == i,:);
			test_label = personID(subset == i,:);

			for j=1:numel(C)
				for k=1:5
					if k ~= i
						vtrain_data = proj_image(subset ~= k,:);
						vtrain_label = personID(subset ~= k,:);
						vtest_data = proj_image(subset == k,:);
						vtest_label = personID(subset == k,:);

						opts = sprintf('-c %f -t 0 -q',C(j));
						[model] = svmtrain(vtrain_label, vtrain_data, opts);
						[~, accuracy, ~] = svmpredict(vtest_label, vtest_data, model,'-q');
						total_accuracy = total_accuracy + accuracy(1);
					end
				end
				avg_accuracy = total_accuracy/4;
				if avg_accuracy > max_accuracy
					maxC = C(j);
					max_accuracy = avg_accuracy;
				end
				total_accuracy = 0;
				avg_accuracy = 0;
			end
			opts = sprintf('-c %f -t 0 -q',maxC);
			[model] = svmtrain(train_label, train_data, opts);
			[~, accuracy, ~] = svmpredict(test_label, test_data, model,'-q');
			total_test_accuracy = total_test_accuracy + accuracy(1);
			fprintf('Optimal C with subset %d chosen as test %d\n',i,maxC);
			fprintf('Accuracy with subset  %d chosen as test: %g\n',i,accuracy(1));
		end
		fprintf('Average test accuracy for d=%d %g\n',D(z),total_test_accuracy/5);
	end

	C = [4^3,4^4,4^5,4^6,4^7,4^8,4^9,4^10];
	gamma = [4^-11,4^-10,4^-9,4^-8,4^-7];
	for z=1:4
		total_accuracy = 0;
		avg_accuracy = 0;
		max_accuracy = 0;
		total_test_accuracy = 0;
		
		eigenvecs = pca_fun(X', D(z));
		proj_image = X'*eigenvecs;
		
		for i=1:5
			train_data = proj_image(subset ~= i,:);
			train_label = personID(subset ~= i,:);
			test_data = proj_image(subset == i,:);
			test_label = personID(subset == i,:);
			
			for g=1:numel(gamma)
				for j=1:numel(C)
					for k=1:5
						if k ~= i
							vtrain_data = proj_image(subset ~= k,:);
							vtrain_label = personID(subset ~= k,:);
							vtest_data = proj_image(subset == k,:);
							vtest_label = personID(subset == k,:);

							opts = sprintf('-c %f -g %f -t 2 -q',C(j),gamma(g));
							[model] = svmtrain(vtrain_label, vtrain_data, opts);
							[~, accuracy, ~] = svmpredict(vtest_label, vtest_data, model,'-q');
							total_accuracy = total_accuracy + accuracy(1);
						end
					end
					avg_accuracy = total_accuracy/4;
					if avg_accuracy > max_accuracy
						maxC = C(j);
						maxg = gamma(g);
						max_accuracy = avg_accuracy;
					end
					total_accuracy = 0;
					avg_accuracy = 0;
				end
			end
			opts = sprintf('-c %f -g %f -t 2 -q',maxC,maxg);
			[model] = svmtrain(train_label, train_data, opts);
			[~, accuracy, ~] = svmpredict(test_label, test_data, model,'-q');
			total_test_accuracy = total_test_accuracy + accuracy(1);
			fprintf('Optimal C with subset %d chosen as test %d\n',i,maxC);
			fprintf('Optimal g with subset %d chosen as test %d\n',i,maxg);
			fprintf('Accuracy with subset  %d chosen as test: %g\n',i,accuracy(1));
		end
		fprintf('Average test accuracy for d=%d %g\n',D(z),total_test_accuracy/5);
	end
end

