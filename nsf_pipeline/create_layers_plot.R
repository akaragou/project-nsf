
library(ggplot2)
png(file="Vgg16 Model Accuracy and Correlation with Humans.png",width=1500, height=950, res=150)
neg_trans <- function(x){
  val <- (x-50)
  val <- val/50
  return(val)
}

pos_trans <- function(x){
  val <- (x*50)
  val <- val+50
  return(val)
}

accuracy_result = read.csv('layer_accuracy_results.csv')
correlation_result = read.csv('layer_correlation_results.csv')

Accuracy <- accuracy_result$Accuracy * 100
correlation <- correlation_result$Correlation 
Results <- rep("Caffe Weights", 15)
Caffe_results <- data.frame(layer_numbers, correlation,Accuracy, Results)
combined_resuls <- rbind(Caffe_results)

ggplot(combined_resuls, aes(x=layer_numbers, color = Results)) +
  coord_cartesian(ylim = c(50,100)) + 
  geom_point(aes(y=Accuracy),alpha =0.5) +
  geom_smooth(aes(y=Accuracy),method='lm',se=FALSE, alpha=0.5)+
  theme_bw() +
  theme(axis.text.x=element_text(angle=45, hjust=1),
         plot.background = element_blank()
          ,panel.grid.major = element_blank()
          ,panel.grid.minor = element_blank()) +
  geom_hline(aes(yintercept=77.4),color='grey40',alpha =0.5) +
  geom_hline(aes(yintercept=78.8),color='grey40',linetype='dashed',alpha =0.5) +
  geom_hline(aes(yintercept=76),color='grey40',linetype='dashed',alpha =0.5) +
  geom_text(aes(x = 14, y = 80),color='grey40', label = "Human Accuracy",alpha =0.5) + 
  geom_point(aes(y=pos_trans(correlation)),alpha =0.5) +
 geom_smooth(aes(y=pos_trans(correlation)),method='loess', alpha=0.5,se=FALSE,span=1)+
 
  scale_y_continuous(sec.axis = sec_axis(~neg_trans(.), name = "Spearman's rho Correlation")) +

  scale_x_discrete(limits = layer_numbers,labels=c('conv1_1', 'conv1_2', 'conv2_1', 
                                                   'conv2_2', 'conv3_1', 'conv3_2', 
                                                   'conv3_3', 'conv4_1', 'conv4_2', 
                                                   'conv4_3', 'conv5_1', 'conv5_2', 
                                                   'conv5_3', 'fc6', 'fc7')) +
  xlab('Layer (Increasing Complexity)') +
  ylab('Accuracy (%)') + 
  ggtitle("Vgg16 Model Accuracy and Correlation with Humans") 

dev.off()

