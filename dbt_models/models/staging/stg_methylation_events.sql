{{ config(tags=['individual']) }}

select
    align_id,
    position as methylation_offset,
    'cpg' as context,
    true as methylated
from {{ source('quant', 'quant__cpg_methylated') }}

union all

select
    align_id,
    position as methylation_offset,
    'cpg' as context,
    false as methylated
from {{ source('quant', 'quant__cpg_unmethylated') }}

union all

select
    align_id,
    position as methylation_offset,
    'chg' as context,
    true as methylated
from {{ source('quant', 'quant__chg_methylated') }}

union all

select
    align_id,
    position as methylation_offset,
    'chg' as context,
    false as methylated
from {{ source('quant', 'quant__chg_unmethylated') }}

union all

select
    align_id,
    position as methylation_offset,
    'chh' as context,
    true as methylated
from {{ source('quant', 'quant__chh_methylated') }}

union all

select
    align_id,
    position as methylation_offset,
    'chh' as context,
    false as methylated
from {{ source('quant', 'quant__chh_unmethylated') }}
