function [ S ] = matlab_gbvs( I )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
m = gbvs(I);
S = m.master_map_resized;
end
