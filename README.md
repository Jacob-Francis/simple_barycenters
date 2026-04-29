Initial grouping which is then used in this repo.

The thesis has reorganised slightly the groups.

\begin{table}[H]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.35}

\begin{tabular}{c|p{3.5cm}|p{3cm}|p{6cm}}
\hline
\textbf{Group} & \textbf{Brief Description} & \set{s Included} & \textbf{Detailed Description} \\
\hline

1 & Boundary, rotation and orientation 
& \set{1}--\set{4} 
& Tests sensitivity to zero padding around a boundary, relative orientation to the boundary and rotated observation and forecasts.
\\
2 & Double penalty 
& \set{5} 
& Designed to assess the double penalty effect, where displacement errors are penalised twice in traditional verification metrics. 
\\
3 & Clustering/Barycentres off support 
& \set{6}--\set{8}
& Cases where barycentres form outside the physical support of the input observation or forecast fields, and clustering is re-investigated.
\\
4 & Noise cases 
& \set{9} 
& Includes randomly generated noise added to otherwise structured fields to test robustness. 
\\
5 & Pathological cases 
& \set{10}
& Artificial extreme examples designed to expose potential failures or instabilities in the methods. 
\\
6 & Spatial bias 
& \set{11}--\set{14} 
& Systematic spatial displacement and bias through shrinking or enlarging the forecasted event.
\\
7 & Intensity error (equal mass) 
& \set{15}--\set{17} 
& Fields with intensity errors but conserved total mass, testing sensitivity to redistribution. 
\\
8 & Intensity error (unequal mass) 
& \set{18}--\set{20} 
& Intensity errors with non-conserved mass but fixed spatial support, hence no translation error. 
\\
9 & Extreme events 
& \set{21}, \set{22} 
& High-intensity feature with strong localisation and limited spatial extent contained in a larger low-intensity support.
\\
10 & Multiscale events 
& \set{23}--\set{25}
& Events containing features at multiple spatial scales simultaneously. 
\\
\hline
\end{tabular}

\caption{Classification of experimental case groups. Each group contains sets designed to test specific aspects of spatial forecast verification, including displacement, intensity errors, clustering, and pathological configurations.}
\label{table:case_groups}
\end{table}