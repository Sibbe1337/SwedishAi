"""Add deepseek_key to User model

Revision ID: 7bcf6cff2f62
Revises: a6a4a779eb15
Create Date: 2025-01-24 09:45:27.734714

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7bcf6cff2f62'
down_revision: Union[str, None] = 'a6a4a779eb15'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('deepseek_key', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users', 'deepseek_key')
    # ### end Alembic commands ###
