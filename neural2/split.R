splitSet <- function()
{
    splitdf <- function(dataframeX, dataframeY, seed=NULL) {
        if (!is.null(seed)) set.seed(seed)
        index <- 1:nrow(dataframeX)
        trainindex <- sample(index, trunc(length(index) * 0.8))
        print(trunc(length(index) * 0.8))
        trainsetX <- dataframeX[trainindex, ]
        trainsetY <- dataframeY[trainindex, ]
        testsetX <- dataframeX[-trainindex, ]
        testsetY <- dataframeY[-trainindex, ]
        rownames(trainsetX) <- NULL
        rownames(trainsetY) <- NULL
        rownames(testsetX) <- NULL
        rownames(testsetY) <- NULL
        list(trainsetX=trainsetX, testsetX=testsetX, 
             trainsetY=trainsetY, testsetY=testsetY)
    }
    
    dataX1 <- read.csv("original/trainP6_1.csv", header = FALSE)
    dataX2 <- read.csv("original/trainP6_2.csv", header = FALSE)
    
    dataY1 <- read.csv("original/trainY1.csv", header = FALSE)
    dataY2 <- read.csv("original/trainY2.csv", header = FALSE)
    
    set1 <- splitdf(dataX1, dataY1, 1)
    set2 <- splitdf(dataX2, dataY2, 1)
    
    View(set1[1])
    write.table(x = set1[1], file = "trainP7_1.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set1[2])
    write.table(x = set1[2], file = "validateP7_1.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set1[3])
    write.table(x = set1[3], file = "trainYP2_1.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set1[4])
    write.table(x = set1[4], file = "validateY1.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set2[1])
    write.table(x = set2[1], file = "trainP7_2.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set2[2])
    write.table(x = set2[2], file = "validateP7_2.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set2[3])
    write.table(x = set2[3], file = "trainYP2_2.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
    View(set2[4])
    write.table(x = set2[4], file = "validateY2.csv", sep = ",", 
                row.names = FALSE, col.names = FALSE)
}