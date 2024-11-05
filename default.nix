{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3;
  
in pkgs.mkShell {
  buildInputs = [
    pkgs.just
    pkgs.poetry
    pkgs.postgresql  # Add PostgreSQL Database
  ];

  shellHook = ''
    # SETUP POSTGRES SERVER
    echo "Loading environment variables from .env"
    if [ -f .env ]; then
      export $(cat .env | xargs)
    else
      echo ".env file not found! Exiting."
      exit 1
    fi

    # Define PGDATA and LOGFILE based on environment variables
    export PGDATA=$(pwd)/pgdata
    LOGFILE=$(pwd)/pgdata/postgresql.log

    echo "Initializing PostgreSQL data directory at $PGDATA"
    
    echo "PGDATA: $PGDATA"
    if [ ! -d "$PGDATA" ]; then
        initdb -D "$PGDATA" | tee -a $LOGFILE
    fi
    
    echo "Stopping any existing PostgreSQL server..."
    pg_ctl -D "$PGDATA" stop || echo "No existing server to stop."

    echo "Starting PostgreSQL server..."
    pg_ctl -D "$PGDATA" -l $LOGFILE start

    # Wait for PostgreSQL to start
    sleep 2

    # Set up the test database, role, and tables
    echo "Setting up the test database..."
    # psql -U $POSTGRES_USER -c "CREATE DATABASE $POSTGRES_DB;" || echo "Database $POSTGRES_DB already exists."

    psql -d $POSTGRES_DB <<SQL | tee -a $LOGFILE
    -- Create the conversation table
    CREATE TABLE IF NOT EXISTS conversation (
        message_id VARCHAR PRIMARY KEY,
        conv_id VARCHAR NOT NULL,
        userid VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        name VARCHAR,
        role VARCHAR NOT NULL,
        message TEXT NOT NULL
    );

    -- Create the utterance_token_info table
    CREATE TABLE IF NOT EXISTS utterance_token_info (
        message_id VARCHAR PRIMARY KEY,
        conv_id VARCHAR NOT NULL,
        userid VARCHAR NOT NULL,
        name VARCHAR,
        role VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        ents JSONB
    );

    -- Create the utterance_text_info table
    CREATE TABLE IF NOT EXISTS utterance_text_info (
        message_id VARCHAR PRIMARY KEY,
        conv_id VARCHAR NOT NULL,
        userid VARCHAR NOT NULL,
        name VARCHAR,
        role VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        moderator JSONB,
        mod_label VARCHAR,
        tern_sent JSONB,
        tern_label VARCHAR,
        emo_27 JSONB,
        emo_27_label VARCHAR
    );

    CREATE TABLE IF NOT EXISTS groups (
        group_id TEXT PRIMARY KEY,
        group_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        last_seen_online TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS user_groups (
        user_id TEXT,
        group_id TEXT,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (group_id) REFERENCES groups (group_id),
        PRIMARY KEY (user_id, group_id)
    );

    CREATE INDEX IF NOT EXISTS idx_user_groups_user_id ON user_groups (user_id);
    CREATE INDEX IF NOT EXISTS idx_user_groups_group_id ON user_groups (group_id);


    CREATE ROLE $POSTGRES_USER WITH LOGIN PASSWORD '$POSTGRES_PASSWORD';
    GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON SCHEMA public TO $POSTGRES_USER;

    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $POSTGRES_USER;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $POSTGRES_USER;

    -- If you're using PostgreSQL 14 or later, you can also add these:
    GRANT pg_read_all_data TO $POSTGRES_USER;
    GRANT pg_write_all_data TO $POSTGRES_USER;
    SQL

    echo "PostgreSQL setup complete. Logs can be found at $LOGFILE"
    # FINISH POSTGRES SERVER

    # BUILD TOPOS PROJECT
    echo "Running just build"
    just build

    echo "Starting the topos server..."
    topos run
  '';
}