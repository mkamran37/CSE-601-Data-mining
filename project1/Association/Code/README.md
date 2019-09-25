# CSE-601
Principal Component Analysis and Frequent itemsets generation using Apriori algorithm and data visualization.

# To generate frequent itemsets (Apriori.py),
1. First install all the packages from requirements.txt
2. Run the program as "python Apriori.py"
3. At the input prompt, first enter the filename without the extension.
    3.1 REQUIRES A MAC FOR THE FILEPATH TO BE VALID!! FOR WINDOWS, NEED TO CHANGE THE ENTIRE FILEPATH VARIABLE IN main()
4. Make sure to keep the input file in the "Data" folder
5. Next, input the minimum support (default = 50%)
6. For the output of each length frequent itemset, check the corresponding .txt file (e.g., "length1output.txt")

# To generate rules from frequent itemsets (AssociationRuleGen.py)
1. Follow the above steps by replacing Apriori.py with AssociationRuleGen.py
2. Additional input: minimum confidence
3. For output, check results.txt
    3.1 For each run of the program, make sure to delete results.txt file as the result is appended to the file.