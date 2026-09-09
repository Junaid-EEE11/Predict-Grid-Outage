# Predict Grid Outage

A machine learning project for predicting electrical grid outages using historical outage records, weather data, and geographic features.

## Overview

This project combines data from multiple authoritative sources (EAGLE-I outage records, NOAA weather data, and US Census data) to build predictive models for grid outages. It implements a rigorous data pipeline with strict handling of data leakage and missing values.

## Features

- **Data Integration**: Aggregates outage records, weather information, and census data into a unified panel dataset
- **Temporal Alignment**: Ensures hourly alignment of features with proper prediction cutoff timestamps
- **Geospatial Analysis**: Incorporates county-level geographic relationships and contiguity
- **Rigorous Data Handling**: Implements strict protocols to avoid data leakage and properly handle missing values
- **Train/Test Split**: Organized data splits for reproducible model evaluation
