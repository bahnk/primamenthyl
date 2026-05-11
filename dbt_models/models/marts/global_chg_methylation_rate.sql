{{ config(tags=['individual']) }}

select
    samples.sample,
    cast(chg_methylated.methylated_count as double)
    / nullif(chg_methylated.methylated_count + chg_unmethylated.unmethylated_count, 0)
        as global_chg_methylation_rate
from {{ source('quant', 'quant__samples') }} as samples
cross join (
    select count(*) as methylated_count
    from {{ source('quant', 'quant__chg_methylated') }}
) as chg_methylated
cross join (
    select count(*) as unmethylated_count
    from {{ source('quant', 'quant__chg_unmethylated') }}
) as chg_unmethylated
