require(class)

# Read in data
RawData <- read.csv("D:/Machine Learning/Classification/data.txt", header=FALSE)
x <- data.frame(x1 = RawData$V1, x2 = RawData$V2)
g <- RawData$V3

# Generate points to be classified
px1 <- seq(min(x$x1)-2, max(x$x1)+2, (max(x$x1)-min(x$x1))/50)
px2 <- seq(min(x$x2)-2, max(x$x2)+2, (max(x$x2)-min(x$x2))/50)
xnew <- expand.grid(px1, px2)

# KNN Classification
k <- 10
mod <- knn(x, xnew, g, k, prob = TRUE)

# Probability of each point
prob <- attr(mod, "prob")
prob <- ifelse(mod=="1", prob, 1-prob)
prob <- matrix(prob, length(px1), length(px2))

# Set margin of the plot
par(mar = rep(2,4))

# Plot the decision boundary
contour(px1, px2, prob, levels=0.5, labels="", xlab="", ylab="", 
        main = paste(as.character(k), "-Nearest Neighbor Classifer"), axes=FALSE)
points(x, col = ifelse(g==1, "coral", "cornflowerblue"))
points(xnew, pch=".", cex=1.2, col = ifelse(prob>0.5, "coral", "cornflowerblue"))
box()