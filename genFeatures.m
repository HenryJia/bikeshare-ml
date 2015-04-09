function genFeatures

Xa1 = load("trainPF4_1.csv")(:, 2:end);
Xb1 = load("trainPF4_2.csv")(:, 2:end);

testXa1 = load("testPF3_1.csv")(:, 2:end);
testXb1 = load("testPF3_2.csv")(:, 2:end); 

monthA1 = Xa1(:, 1);
monthB1 = Xb1(:, 1);

monthA2 = Xa1(:, 2);
monthB2 = Xb1(:, 2);

monthA3 = Xa1(:, 3);
monthB3 = Xb1(:, 3);

monthA4 = Xa1(:, 4);
monthB4 = Xb1(:, 4);

dayA1 = Xa1(:, 5);
dayB1 = Xb1(:, 5);

timeA1 = Xa1(:, 6);
timeB1 = Xb1(:, 6);

timeA2 = Xa1(:, 7);
timeB2 = Xb1(:, 7);

timeA3 = Xa1(:, 8);
timeB3 = Xb1(:, 8);

timeA4 = Xa1(:, 9);
timeB4 = Xb1(:, 9);

timeA5 = Xa1(:, 10);
timeB5 = Xb1(:, 10);

seasonA1 = Xa1(:, 11);
seasonB1 = Xb1(:, 11);

holidayA1 = Xa1(:, 12);
holidayB1 = Xb1(:, 12);

workingdayA1 = Xa1(:, 13);
workingdayB1 = Xb1(:, 13);

weatherA1 = Xa1(:, 14);
weatherB1 = Xb1(:, 14);

tempA1 = Xa1(:, 15);
tempB1 = Xb1(:, 15);

atempA1 = Xa1(:, 16);
atempB1 = Xb1(:, 16);

humidityA1 = Xa1(:, 17);
humidityB1 = Xb1(:, 17);

windspeedA1 = Xa1(:, 18);
windspeedA1 = Xa1(:, 18);


