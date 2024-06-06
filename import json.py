import json

def json_to_sql(json_file_path, sql_file_path, table_name):
    # Step 1: Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
 
    if isinstance(data, list):
        sample_record = data[0]
    elif isinstance(data, dict):
        sample_record = data
        data = [data]  
    else:
        raise ValueError("Unsupported JSON structure")

   
    columns = sample_record.keys()
    create_table_statement = f"CREATE TABLE {table_name} (\n"
    create_table_statement += ",\n".join([f"    {col} TEXT" for col in columns])
    create_table_statement += "\n);"
    
    insert_statements = []
    for record in data:
        values = [f"'{str(record[col]).replace('\'', '\'\'')}'" for col in columns]
        insert_statement = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
        insert_statements.append(insert_statement)
  
    with open(sql_file_path, 'w') as file:
        file.write(create_table_statement + "\n")
        file.write("\n".join(insert_statements))

# Usage example
json_file_path = 'data.json'
sql_file_path = 'data.sql'
table_name = 'my_table'
json_to_sql(json_file_path, sql_file_path, table_name)