import json

def json_to_sql(json_file_path, sql_file_path, table_name):
    # Step 1: Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Step 2: Parse the JSON data
    if isinstance(data, list):
        sample_record = data[0]
    elif isinstance(data, dict):
        sample_record = data
        data = [data]  # Convert to list for consistent processing
    else:
        raise ValueError("Unsupported JSON structure")

    # Step 3: Generate SQL INSERT statements
    columns = sample_record.keys()
    insert_statements = []
    for record in data:
        values = [f"'{str(record[col]).replace('\'', '\'\'')}'" for col in columns]
        insert_statement = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});"
        insert_statements.append(insert_statement)
    
    # Step 4: Write to SQL script file
    with open(sql_file_path, 'w') as file:
        file.write("\n".join(insert_statements))

def convert_files(json_file_paths, sql_file_paths, table_names):
    for i in range(len(json_file_paths)):
        json_to_sql(json_file_paths[i], sql_file_paths[i], table_names[i])

# json_file_paths = ["appointments.json", "dependents.json", "eventAttendees.json", "events.json", "posts.json", "users.json", "volunteers.json"]
# sql_file_paths = ["appointments.sql", "dependents.sql", "eventAttendees.sql", "events.sql", "posts.sql", "users.sql", "volunteers.sql"]
# table_names = "appiointments", "dependents", "eventAttendees", "events", "posts", "users", "volunteers"


def main ():

    # convert_files(json_file_paths, sql_file_paths, table_names)
    json_to_sql("posts (1).json", "posts.sql", "posts")

main()