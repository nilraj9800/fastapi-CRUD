"""update User table

Revision ID: 49a033b8177b
Revises: 
Create Date: 2022-04-24 00:32:31.539621

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '49a033b8177b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    #test code only use if needed
    op.add_column('user',sa.Column('test',sa.String(),nullable=False))

def downgrade():
    pass
