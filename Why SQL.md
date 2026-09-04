# Software Information

I used "SQL*Plus: Release 11.2.0.2.0" with "Oracle Database 11g Express Edition (XE)"

It is quite an old version, I used it in my studies of DBMS and am comfortable with it so didn't change it.

# The Reason

I needed SQL to handle some data rearrangement and splitting tasks, for an initial idea.
The dataset is large to load into memeory so I can't do it all with pandas or polars.

I will add an equivalent python file here which will implement an External Sort mechanism like Oracle does to perform Temporal Sorts, GeoSpacial Sorts and even Geo-Temporal-Chunking of my data.
For the time being I will perform it via SQL and provide the Spool files here (after editing the paths.)