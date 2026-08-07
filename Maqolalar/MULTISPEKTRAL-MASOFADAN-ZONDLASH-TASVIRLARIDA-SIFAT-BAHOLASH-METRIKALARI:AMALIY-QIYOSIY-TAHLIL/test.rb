users = a list of users (objects with attributes like name, age, etc.)
whitelisted_emails = a list of email addresses that are allowed to access the system

whitelisted_emails = ['a1, a2, a3, a4, a5']
users.each do |user|
    if whitelisted_emails.include?(user.email)
        puts "Access granted to #{user.name}"
    else
        puts "Access denied to #{user.name}"
    end
end
2*N+1

O (N x M)



whitelisted_emails = {
    'email@gm.co' => true
}


users.each do |user|
    if whitelisted_emails[user.email]
        puts "Access granted to #{user.name}"
    else
        puts "Access denied to #{user.name}"
    end
end
2 * N