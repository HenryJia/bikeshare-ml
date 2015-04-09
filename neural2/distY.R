distY <- function()
{
    Y = read.csv(file = "trainYP2_1.csv", header = FALSE)
    hist(x = table(Y), breaks = 100)
}