{{ config(tags=['individual']) }}

select
    samples.sample,
    cast(chh_methylated.methylated_count as double)
    / nullif(chh_methylated.methylated_count + chh_unmethylated.unmethylated_count, 0)
        as global_chh_methylation_rate
from {{ source('quant', 'quant__samples') }} as samples
cross join (
    select count(*) as methylated_count
    from {{ source('quant', 'quant__chh_methylated') }}
) as chh_methylated
cross join (
    select count(*) as unmethylated_count
    from {{ source('quant', 'quant__chh_unmethylated') }}
) as chh_unmethylated
