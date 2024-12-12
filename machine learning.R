rm(list = ls())
#
work.path <- "E:\\bioinformatics\\machine learning"
setwd(work.path) 

#
code.path <- file.path(work.path, "Codes")
data.path <- file.path(work.path, "InputData")
res.path <- file.path(work.path, "Results")
fig.path <- file.path(work.path, "Figures")

#
if (!dir.exists(data.path)) dir.create(data.path)
if (!dir.exists(res.path)) dir.create(res.path)
if (!dir.exists(fig.path)) dir.create(fig.path)
if (!dir.exists(code.path)) dir.create(code.path)

#
library(openxlsx)
library(seqinr)
library(plyr)
library(randomForestSRC)
library(glmnet)
library(plsRglm)
library(gbm)
library(caret)
library(mboost)
library(e1071)
library(BART)
library(MASS)
library(snowfall)
library(xgboost)
library(ComplexHeatmap)
library(RColorBrewer)
library(pROC)

#
source(file.path(code.path, "ML.R"))

FinalModel <- c("panML", "multiLogistic")[2]

#training set
#
Train_expr <- read.table(file.path(data.path, "Training_expr.txt"), 
                         header = T, sep = "\t", row.names = 1,
                         check.names = F,stringsAsFactors = F)
#
Train_class <- read.table(file.path(data.path, "Training_class.txt"), 
                          header = T, sep = "\t", row.names = 1,
                          check.names = F,stringsAsFactors = F)
#
comsam <- intersect(rownames(Train_class), colnames(Train_expr))

Train_expr <- Train_expr[,comsam]
Train_class <- Train_class[comsam,,drop = F]

#
Test_expr <- read.table(file.path(data.path, "Testing_expr.txt"), 
                        header = T, sep = "\t", row.names = 1,
                        check.names = F,stringsAsFactors = F)
Test_class <- read.table(file.path(data.path, "Testing_class.txt"), 
                         header = T, sep = "\t", row.names = 1,
                         check.names = F,stringsAsFactors = F)
comsam <- intersect(rownames(Test_class), colnames(Test_expr))
Test_expr <- Test_expr[,comsam]
Test_class <- Test_class[comsam,,drop = F]

#
comgene <- intersect(rownames(Train_expr),rownames(Test_expr))
#
Train_expr <- t(Train_expr[comgene,])
Test_expr <- t(Test_expr[comgene,])

Train_set = scaleData(data = Train_expr, centerFlags = T, scaleFlags = T)
names(x = split(as.data.frame(Test_expr), f = Test_class$Cohort))

Test_set = scaleData(data = Test_expr, cohort = Test_class$Cohort, 
                     centerFlags = T, scaleFlags = T)

#
methods <- read.xlsx(file.path(code.path, "methods.xlsx"), startRow = 2)
methods <- methods$Model
methods <- gsub("-| ", "", methods)

#——————————————
classVar = "outcome"
min.selected.var = 0

#try
Variable = colnames(Train_set)
preTrain.method =  strsplit(methods, "\\+") 
preTrain.method = lapply(preTrain.method, function(x) rev(x)[-1]) 
preTrain.method = unique(unlist(preTrain.method))

preTrain.var <- list() 
set.seed(seed = 777)
for (method in preTrain.method){
  preTrain.var[[method]] = RunML(method = method, 
                                 Train_set = Train_set, 
                                 Train_label = Train_class,
                                 mode = "Variable", 
                                 classVar = classVar) 
}

preTrain.var[["simple"]] <- colnames(Train_set)

#
model <- list()
set.seed(seed = 777)
Train_set_bk = Train_set
for (method in methods){
  cat(match(method, methods), ":", method, "\n")
  method_name = method 
  method <- strsplit(method, "\\+")[[1]]
  
  if (length(method) == 1) method <- c("simple", method) 
  
  Variable = preTrain.var[[method[1]]] 
  Train_set = Train_set_bk[, Variable] 
  Train_label = Train_class            
  model[[method_name]] <- RunML(method = method[2],    
                                Train_set = Train_set,   
                                Train_label = Train_label, 
                                mode = "Model",        
                                classVar = classVar) 
  if(length(ExtractVar(model[[method_name]])) <= min.selected.var) {
    model[[method_name]] <- NULL
  }
}

Train_set = Train_set_bk
rm(Train_set_bk)
saveRDS(model, file.path(res.path, "model.rds"))

if (FinalModel == "multiLogistic"){
  logisticmodel <- lapply(model, function(fit){ 
    tmp <- glm(formula = Train_class[[classVar]] ~ .,
               family = "binomial", 
               data = as.data.frame(Train_set[, ExtractVar(fit)]))
    tmp$subFeature <- ExtractVar(fit) 
    return(tmp)
  })
}
saveRDS(logisticmodel, file.path(res.path, "logisticmodel.rds"))

#evalue
model <- readRDS(file.path(res.path, "model.rds"))
methodsValid <- names(model)


RS_list <- list()
for (method in methodsValid){
  RS_list[[method]] <- CalPredictScore(fit = model[[method]], 
                                       new_data = rbind.data.frame(Train_set,Test_set)) 
}
RS_mat <- as.data.frame(t(do.call(rbind, RS_list)))
write.table(RS_mat, file.path(res.path, "RS_mat.txt"),sep = "\t", row.names = T, col.names = NA, quote = F)

Class_list <- list()
for (method in methodsValid){
  Class_list[[method]] <- PredictClass(fit = model[[method]], 
                                       new_data = rbind.data.frame(Train_set,Test_set)) 
}
Class_mat <- as.data.frame(t(do.call(rbind, Class_list)))

write.table(Class_mat, file.path(res.path, "Class_mat.txt"), 
            sep = "\t", row.names = T, col.names = NA, quote = F)

fea_list <- list()
for (method in methodsValid) {
  fea_list[[method]] <- ExtractVar(model[[method]])
}

fea_df <- lapply(model, function(fit){
  data.frame(ExtractVar(fit))
})
fea_df <- do.call(rbind, fea_df)
fea_df$algorithm <- gsub("(.+)\\.(.+$)", "\\1", rownames(fea_df))
colnames(fea_df)[1] <- "features"
write.table(fea_df, file.path(res.path, "fea_df.txt"),
            sep = "\t", row.names = F, col.names = T, quote = F)

AUC_list <- list()
for (method in methodsValid){
  AUC_list[[method]] <- RunEval(fit = model[[method]],     
                                Test_set = Test_set,     
                                Test_label = Test_class,  
                                Train_set = Train_set,   
                                Train_label = Train_class, 
                                Train_name = "GSE59071",     
                                cohortVar = "Cohort",    
                                classVar = classVar)   
}
AUC_mat <- do.call(rbind, AUC_list)
write.table(AUC_mat, file.path(res.path, "AUC_mat.txt"),
            sep = "\t", row.names = T, col.names = T, quote = F)

#plot
AUC_mat <- read.table(file.path(res.path, "AUC_mat.txt"),sep = "\t", row.names = 1, header = T,check.names = F,stringsAsFactors = F)
avg_AUC <- apply(AUC_mat, 1, mean)           
avg_AUC <- sort(avg_AUC, decreasing = T)    
AUC_mat <- AUC_mat[names(avg_AUC), ]     
fea_sel <- fea_list[[rownames(AUC_mat)[1]]] 
avg_AUC <- as.numeric(format(avg_AUC, digits = 3, nsmall = 3))

if(ncol(AUC_mat) < 3) { 
  CohortCol <- c("red","blue") 
} else { 
  CohortCol <- brewer.pal(n = ncol(AUC_mat), name = "Paired")
}
names(CohortCol) <- colnames(AUC_mat)

cellwidth = 1; cellheight = 0.5
hm <- SimpleHeatmap(AUC_mat, 
                    avg_AUC, 
                    CohortCol, "steelblue",
                    cellwidth = cellwidth, cellheight = cellheight, 
                    cluster_columns = F, cluster_rows = F) 

pdf(file.path(fig.path, "AUC.pdf"), 
    width = cellwidth * ncol(AUC_mat) + 3, 
    height = cellheight * nrow(AUC_mat) * 0.45)
draw(hm)
invisible(dev.off())
