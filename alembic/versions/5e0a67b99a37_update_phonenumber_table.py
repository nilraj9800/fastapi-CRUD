"""update phonenumber table

Revision ID: 5e0a67b99a37
Revises: c7917d8cc241
Create Date: 2022-04-24 15:57:43.019885

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5e0a67b99a37'
down_revision = 'c7917d8cc241'
branch_labels = None
depends_on = None


def upgrade():
    #update the phonenumber table
    pass

def downgrade():
    pass
