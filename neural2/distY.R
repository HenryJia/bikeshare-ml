distY <- function()
{
    Y = read.csv(file = "trainYP2_1.csv", header = FALSE)
    td = data.frame(table(Y))
    hist(x = table(Y), breaks = 100)
    lines(x = 1:max(Y), y = (2000 / (1:max(Y) + 20)))
}