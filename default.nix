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
    psql -d $POSTGRES_DB <<SQL | tee -a $LOGFILE
    CREATE TABLE IF NOT EXISTS conversation_cache (
        conv_id TEXT PRIMARY KEY,
        message_data JSONB NOT NULL
    );

    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        properties JSONB
    );

    CREATE TABLE IF NOT EXISTS relations (
        source_id TEXT,
        relation_type TEXT,
        target_id TEXT,
        properties JSONB,
        PRIMARY KEY (source_id, relation_type, target_id)
    );

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